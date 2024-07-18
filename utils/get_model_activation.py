import torch as t

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

    with t.no_grad(): model(input)
    for handle in handles: handle.remove()

    return activation

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