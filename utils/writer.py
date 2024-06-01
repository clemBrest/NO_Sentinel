from pytorch_lightning.utilities.model_summary import ModelSummary, LayerSummary
import os

"""
This module contains the function summary_file that writes a summary of the model to a file.
It also prints the summary to the console if print_flag is set to True.
"""

def summary_file(args, Lmodel, train_data, test_data, print_flag=False):
    # Ensure the saving directory exists
    str_name = args.str_name
    os.makedirs(args.saving_path+'/'+ str_name , exist_ok=True)

    with open(args.saving_path + '/'+ str_name + '/model_summary.txt', 'w') as f:
        #title
        f.write('       Model Summary\n\n')

        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write('\n')

        # Write ModelSummary to file
        f.write(str(ModelSummary(Lmodel)) + '\n\n')
        f.write(str(Lmodel.model) + '\n\n')

        # Loop through layers and write layer summaries
        for layer in Lmodel.model.children():
            lSum = LayerSummary(layer)
            f.write(f"{lSum.layer_type}: {lSum.num_parameters} parameters\n")
        f.write('\n')

        # write some data info
        f.write(f"Training data shape: {train_data.__len__()}\n")  # Added missing newline
        f.write(f"Test data shape: {test_data.__len__()}")

        #print this file
    if print_flag:
        with open(args.saving_path + '/'+ str_name+ '/model_summary.txt', 'r') as f:
            print(f.read())