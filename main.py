import pysindy as ps
from pysindy.differentiation import FiniteDifference   
from pysindy.optimizers import STLSQ
from pysindy.feature_library import CustomLibrary, polynomial_library
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def load_data():
    
    data = scipy.io.loadmat('scalini.mat')
    # print(len(data['data032']))
    eval_data = scipy.io.loadmat('seni_nooffset.mat')    #changing this file will change also the eval_data keys so you have to modify also in the lines 27
    t = np.array(data['data032'][0][0:20000])           # 29 31 33 from data038 (for seni_nooffset.mat) to data042 for 'seni_offset2.mat' and to data044 
    t = t.reshape(len(t),)                              #for 'seni_largeamplitude_noaffset.mat'
    u = np.array(data['data032'][1][0:20000])
    u = u.reshape(len(u),1)
    x1 = np.array(data['data032'][2][0:20000])
    x1 = x1.reshape(len(x1),1)
    # print(x.shape)
    v = np.array(data['data032'][3][0:20000])
    v = v.reshape(len(v),1)
    x = np.concatenate((x1, v), axis=1)
    
    print(eval_data.keys())
    eval_t = np.array(eval_data['data038'][0][0:20000])
    eval_t = eval_t.reshape(len(eval_t),)
    eval_u = np.array(eval_data['data038'][1][0:20000])
    eval_u = eval_u.reshape(len(eval_u),1)
    eval_x1 = np.array(eval_data['data038'][2][0:20000])
    eval_x1 = eval_x1.reshape(len(eval_x1),1)
    eval_v = np.array(eval_data['data038'][3][0:20000])
    eval_v = eval_v.reshape(len(eval_v),1)

    eval_x = np.concatenate((eval_x1, eval_v), axis=1)

    return  data, t, u, x, eval_t, eval_u, eval_x

def plot_data(eval_x : np.ndarray,  pred_x: np.ndarray, title, legend1, legend2):
        plt.plot(eval_x, c='b')
        plt.plot(pred_x, c='r')
        plt.title(title)
        plt.legend([legend1,legend2])
        plt.show()

def fit(x : np.ndarray, t : np.ndarray, u, feature_library, optimizer : str, differentiation_method : str) -> ps.SINDy:
    if optimizer.capitalize == 'STLSQ':
        optimizer = STLSQ(threshold=0.025, max_iter=20)
    else:
        optimizer = STLSQ(threshold=0.01, max_iter=20)             
    
    if differentiation_method.capitalize == 'FINITEDIFFERENCE':
        differentiation_method = FiniteDifference()
    else:
        differentiation_method = FiniteDifference()

    model = ps.SINDy(optimizer=optimizer, 
                    differentiation_method=differentiation_method,
                    # feature_library=feature_library                 uncomment to use a customized library if not specified is used the polinomial library
                    feature_names=['x','xd'],
                    discrete_time=False)
    model.fit(x,t,u=u)
    print(f'xi: {model.coefficients().T}')
    # model.print()

    return model

def compute_trajectory(model : ps.SINDy, x0: np.ndarray, t_eval, u_eval) -> np.ndarray:
    predicted_x = model.simulate(x0, t=t_eval, u=u_eval)
    return predicted_x

# function to create customized libraries 
def create_custom_library():
    functions =  [lambda x,y: x*np.exp(x),
                  lambda y: y,
                  lambda u,y: u/y]              #lambda x,y: x/y*np.exp(x),  
    lib_custom = CustomLibrary(library_functions=functions)
    return lib_custom

def main():
    data, t, u, x, eval_t, eval_u, eval_x  = load_data()
    
    model = fit(x, t, u, polynomial_library, 'STLSQ', 'finitedifference')
    x0 = np.array([0,0])
    predicted_x = compute_trajectory(model, x0, eval_t, u_eval=eval_u)
    # print(predicted_x.shape)
    #plots of evaluation data 
    plot_data(eval_x[:,0], predicted_x[:,0],title='position',legend1='real_pos',legend2='pred_pos') 
    plot_data(eval_x[:,1], predicted_x[:,1],title='velocity',legend1='real_vel',legend2='pred_vel')
    #plots of training data
    predicted_x2 = compute_trajectory(model, x0, t, u)
    plot_data(x[:,0], predicted_x2[:,0],title='training position',legend1='real_pos',legend2='pred_pos') 
    plot_data(x[:,1], predicted_x2[:,1],title='training velocity',legend1='real_vel',legend2='pred_vel')

main()

