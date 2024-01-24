from pickle import load

def get_model(model_path: str):
    with open(model_path, 'rb') as f:
        model = load(f)
    return model