# qi_utils.py
import torch
import cmath
import logging

# Setup basic logging
logger = logging.getLogger(__name__)
if not logger.handlers: # Avoid adding multiple handlers if reloaded
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# SCALE_EPSILON might still be needed if tan inputs get too close to +/- pi/2 after scaling
SCALE_EPSILON_TAN = 1e-7

def check_tensor_stats(tensor, name="tensor", enabled=False):
    if not enabled:
        return
    if isinstance(tensor, dict): # Handle state_dict
        for k, v in tensor.items():
            check_tensor_stats(v, f"{name}.{k}", enabled)
        return

    if not isinstance(tensor, torch.Tensor):
        logger.warning(f"check_tensor_stats: {name} is not a tensor, it's a {type(tensor)}")
        return

    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan or has_inf:
        logger.warning(f"Stats for {name}: min={tensor.min().item():.4e}, max={tensor.max().item():.4e}, mean={tensor.mean().item():.4e}, std={tensor.std().item():.4e}, HAS_NAN={has_nan}, HAS_INF={has_inf}")
    else:
        logger.debug(f"Stats for {name}: min={tensor.min().item():.4e}, max={tensor.max().item():.4e}, mean={tensor.mean().item():.4e}, std={tensor.std().item():.4e}")


def real_to_phase_atan(real_tensor):
    """Maps a real tensor to phases in [0, 2*pi) using atan."""
    # torch.atan maps R to (-pi/2, pi/2)
    atan_output = torch.atan(real_tensor)
    # Scale to (0, pi)
    scaled_to_0_pi = atan_output + (cmath.pi / 2)
    # Scale to (0, 2*pi)
    phases = scaled_to_0_pi * 2
    return phases

def phase_to_real_tan(phase_tensor):
    """Maps phases from [0, 2*pi) back to the real domain using tan."""
    # Scale from [0, 2*pi) to [0, pi)
    scaled_to_0_pi = phase_tensor / 2
    # Scale to (-pi/2, pi/2)
    atan_input_range = scaled_to_0_pi - (cmath.pi / 2)
    
    # Clamp to avoid tan(pi/2) or tan(-pi/2)
    clamped_atan_input = torch.clamp(atan_input_range, 
                                     -(cmath.pi / 2) + SCALE_EPSILON_TAN, 
                                     (cmath.pi / 2) - SCALE_EPSILON_TAN)
    real_tensor = torch.tan(clamped_atan_input)
    return real_tensor

def encode_params(real_params_dict, secret_key_phase, debug_stats=False):
    encoded_dict = {}
    if debug_stats: logger.debug(f"Encoding with secret_key_phase: {secret_key_phase:.4f}")
    check_tensor_stats(real_params_dict, "real_params_dict_before_encode", enabled=debug_stats)

    for name, param_tensor in real_params_dict.items():
        if not isinstance(param_tensor, torch.Tensor):
            logger.error(f"Item {name} in real_params_dict is not a tensor: {type(param_tensor)}")
            continue
        
        param_tensor_cpu = param_tensor.cpu().float() # Ensure float for atan
        check_tensor_stats(param_tensor_cpu, f"{name}_real_cpu", enabled=debug_stats)

        phases = real_to_phase_atan(param_tensor_cpu)
        check_tensor_stats(phases, f"{name}_phases_before_rotation", enabled=debug_stats)
        
        rotated_phases = phases + secret_key_phase
        check_tensor_stats(rotated_phases, f"{name}_phases_after_rotation", enabled=debug_stats)
        
        # torch.polar expects magnitudes and angles. Magnitudes are 1 for phase encoding.
        encoded_dict[name] = torch.polar(torch.ones_like(rotated_phases), rotated_phases)
        check_tensor_stats(encoded_dict[name], f"{name}_encoded_complex", enabled=debug_stats)
        
    return encoded_dict

def decode_params(encoded_params_dict, secret_key_phase, debug_stats=False):
    decoded_dict = {}
    if debug_stats: logger.debug(f"Decoding with secret_key_phase: {secret_key_phase:.4f}")
    check_tensor_stats(encoded_params_dict, "encoded_params_dict_before_decode", enabled=debug_stats)

    for name, complex_tensor in encoded_params_dict.items():
        if not isinstance(complex_tensor, torch.Tensor) or not complex_tensor.is_complex():
            logger.error(f"Item {name} in encoded_params_dict is not a complex tensor: {type(complex_tensor)}")
            continue

        complex_tensor_cpu = complex_tensor.cpu()
        check_tensor_stats(complex_tensor_cpu, f"{name}_complex_cpu", enabled=debug_stats)

        # Unrotate: multiply by e^(-i * secret_key_phase)
        unrotator_phase = -secret_key_phase
        unrotator = torch.polar(torch.tensor(1.0, dtype=torch.float32), torch.tensor(unrotator_phase, dtype=torch.float32))
        unrotated_complex_repr = complex_tensor_cpu * unrotator.to(complex_tensor_cpu.device) # Ensure same device
        check_tensor_stats(unrotated_complex_repr, f"{name}_unrotated_complex", enabled=debug_stats)
        
        phases_from_complex = torch.angle(unrotated_complex_repr) # Range: (-pi, pi]
        check_tensor_stats(phases_from_complex, f"{name}_phases_from_angle", enabled=debug_stats)

        # Normalize phase to [0, 2*pi) as expected by phase_to_real_tan
        normalized_phases = (phases_from_complex + 2 * cmath.pi) % (2 * cmath.pi)
        check_tensor_stats(normalized_phases, f"{name}_normalized_phases", enabled=debug_stats)
        
        decoded_real_tensor = phase_to_real_tan(normalized_phases)
        check_tensor_stats(decoded_real_tensor, f"{name}_decoded_real", enabled=debug_stats)
        decoded_dict[name] = decoded_real_tensor
        
    return decoded_dict

def aggregate_encoded_params(client_encoded_dicts_list, debug_stats=False):
    if not client_encoded_dicts_list:
        logger.warning("aggregate_encoded_params: No client dicts to aggregate.")
        return {}

    # Summation
    aggregated_sum_dict = {}
    # Find a valid template from the list
    template_dict = None
    for client_dict_item in client_encoded_dicts_list:
        if client_dict_item and isinstance(client_dict_item, dict) and len(client_dict_item) > 0:
            template_dict = client_dict_item
            break
    
    if template_dict is None:
        logger.warning("aggregate_encoded_params: No valid template client_dict found.")
        return {}

    for name in template_dict.keys():
        # Ensure accumulation is done with complex type
        param_template = template_dict[name]
        if isinstance(param_template, torch.Tensor) and param_template.is_complex():
             aggregated_sum_dict[name] = torch.zeros_like(param_template, dtype=torch.complex64)
        else:
            logger.warning(f"aggregate_encoded_params: Template parameter {name} is not a complex tensor. Type: {type(param_template)}. Skipping.")


    active_clients_count = 0
    for client_dict in client_encoded_dicts_list:
        if client_dict and isinstance(client_dict, dict): # Ensure client actually provided params
            active_clients_count +=1
            for name, complex_tensor in client_dict.items():
                if name in aggregated_sum_dict:
                    if isinstance(complex_tensor, torch.Tensor) and complex_tensor.is_complex():
                        aggregated_sum_dict[name] += complex_tensor.cpu() # Ensure CPU for aggregation
                    else:
                        logger.warning(f"aggregate_encoded_params: Client tensor for {name} is not complex. Skipping.")
                # else: # Parameter not in template (e.g. from partial client update, or inconsistent models)
                #     logger.warning(f"aggregate_encoded_params: Parameter '{name}' from a client not in aggregation template. Skipping.")
    
    if active_clients_count == 0:
        logger.warning("aggregate_encoded_params: No valid encoded parameters to aggregate from active clients.")
        return template_dict if template_dict else {} # Return template (or empty) if no contributions

    # Averaging
    averaged_encoded_dict = {}
    for name, summed_tensor in aggregated_sum_dict.items():
        averaged_encoded_dict[name] = summed_tensor / active_clients_count
    
    if debug_stats: logger.debug("Aggregation complete.")
    check_tensor_stats(averaged_encoded_dict, "averaged_encoded_dict_after_aggregate", enabled=debug_stats)
    return averaged_encoded_dict

def test_encode_decode_identity(real_params_dict, secret_key_phase, atol=1e-4):
    logger.info(f"--- Testing encode-decode identity with secret_key_phase: {secret_key_phase:.4f} ---")
    encoded = encode_params(real_params_dict, secret_key_phase, debug_stats=True)
    decoded = decode_params(encoded, secret_key_phase, debug_stats=True)
    
    all_close = True
    max_overall_diff = 0.0
    for name in real_params_dict.keys():
        original = real_params_dict[name].cpu().float()
        reconstructed = decoded.get(name) # Use .get for safety
        if reconstructed is None:
            logger.error(f"Test_encode_decode: Decoded param {name} is None.")
            all_close = False
            continue
            
        reconstructed = reconstructed.float()

        if not torch.allclose(original, reconstructed, atol=atol, rtol=1e-3): # Added rtol
            current_max_diff = (original - reconstructed).abs().max().item()
            max_overall_diff = max(max_overall_diff, current_max_diff)
            logger.warning(f"Test_encode_decode: Mismatch for param {name}. Max diff: {current_max_diff:.4e}")
            # logger.debug(f"Original sample for {name}: {original.view(-1)[:5]}")
            # logger.debug(f"Reconstructed sample for {name}: {reconstructed.view(-1)[:5]}")
            all_close = False
    if all_close:
        logger.info(f"Encode-decode identity test PASSED (within tolerance atol={atol}). Max overall diff observed (where not allclose): {max_overall_diff:.4e}")
    else:
        logger.error(f"Encode-decode identity test FAILED. Max overall diff observed: {max_overall_diff:.4e}")
    return decoded