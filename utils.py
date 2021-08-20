import os
import matplotlib.pyplot as plt
import pickle



PROJECT_ROOT_DIR = '.'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, "model")
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data")


def save_fig(fig_id, tight_layout=True, fig_extension="eps", resolution=600):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def model_cache(compiled_model, model_name, **kwargs):
    path = os.path.join(MODEL_PATH, model_name + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(compiled_model, f)
    print("Model cached as:" + model_name + '.pkl')


def model_load(model_name):
    path = os.path.join(MODEL_PATH, model_name + '.pkl')
    try:
        sm = pickle.load(open(path, 'rb'))
    except:
        raise FileNotFoundError
    else:
        print("Using cached Model:" + model_name)
    return sm
