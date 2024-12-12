import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch


def plot(vec, plot_type):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    
    clear_output(True)
    plt.figure(figsize=(20, 5))
    
    # Set the title and plot based on the plot type
    plt.subplot(131)
    if plot_type == 'reward':
        plt.title(f'Epoch {len(vec)}. Reward: {vec[-1]:.2f}')
    elif plot_type == 'loss':
        plt.title(f'Epoch {len(vec)}. Loss: {vec[-1]:.2f}')
    
    plt.plot(vec)
    plt.xlabel('Epochs')
    plt.ylabel(plot_type.capitalize())
    plt.grid(True)
    plt.show()


