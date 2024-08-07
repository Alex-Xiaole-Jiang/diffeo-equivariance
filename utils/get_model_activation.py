import torch as t
from .diffeo_container import diffeo_container

def retrieve_layer_activation(model, input, layer_index):
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    handles = []

    if len(input) == 3: input = input[None, :, :, :]

    # layers = list(model.children())
    # layers_flat = flatten(layers)
    if hasattr(model, 'encoder'):
        layers_flat = get_flatten_children(model.encoder)
    else:
        layers_flat = get_flatten_children(model)

    for index in layer_index:
        handles.append(layers_flat[index - 1].register_forward_hook(getActivation(str(index))))

    with t.no_grad(): result = model(input)
    for handle in handles: handle.remove()

    return activation, result

def get_flatten_children(model):
    return flatten(list(model.children()))

def flatten(array):
    result = []
    for element in array:
        if hasattr(element, "__iter__"):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


def inv_diff_hook(inverse_diffeo):
    if not isinstance(inverse_diffeo, diffeo_container):        
        raise Exception('diffeo is not a diffeo_container')
    def hook(module, input, output):
        return inverse_diffeo(output, in_inference=True)
    return hook