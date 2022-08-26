from pathlib import Path
path_model_Lor = Path('data/Lorenz_Atractor/')
# path_model_Lin = 'Simulations/Linear/'

#path_model = path_model_Lin
path_model = path_model_Lor

path_session = Path.joinpath(path_model, Path('covariance_experiment/'))
