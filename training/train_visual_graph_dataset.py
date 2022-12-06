import os
import sys
import time
import pathlib
import warnings
import shutil
import random

import click
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import roc_auc_score

PATH = pathlib.Path(__file__).parent.absolute()
HYPER_PATH = os.path.join(PATH, 'hyper', 'hyper_vgd_mock.py')


@click.command()
@click.option('--model', type=click.STRING, default='MEGAN',
              help='The base model to be trained to predict the target values.')
@click.option('--xai-method', type=click.STRING, default=None,
              help='The XAI method to be used to create the explanations for the given model. If a '
                   'self-explaining model is used, this option can be ignored')
@click.option('--dataset', type=click.STRING, default='VgdMockDataset',
              help='Name of the dataset to be used for training.')
@click.option('--hyper', type=click.STRING, default=HYPER_PATH,
              help='Filepath to the hyperparameter config file to be used.')
@click.option('--make', type=click.STRING, default='make_model',
              help='Name of the "make" function or model to use')
@click.option('--gpu', type=click.INT, default=0,
              help='GPU index used for training')
@click.option('--fold', type=click.INT, default=0,
              help='Split or fold indices to run')
@click.option('--show-warnings', is_flag=True,
              help='Will suppress tensorflow warning by default. Use this flag to show them')
@click.option('--visualization-ratio', type=click.FloatRange(0, 1), default=1.0,
              help='The ratio of how many samples from the test set should be used to visualize the '
                   'generated explanations on')
def main(model: str,
         xai_method: str,
         dataset: str,
         hyper: str,
         make: str,
         gpu: int,
         fold: int,
         show_warnings: bool,
         visualization_ratio: float):
    """
    Train a model of choice on a "Visual Graph Dataset" (VGD).

    Such a VGD is a special dataset
    for *Explainable AI* (XAI). Besides the main prediction target, such VGDs are also evaluated for
    different explainability metrics, such as:

    * Explanation Accuracy: If the dataset contains ground truth explanations they are compared with
      the explanations generated by the model.

    * Sparsity: Good explanations should be sparse. This value between 0 and 1 illustrates the percentage
      of the input elements which have been marked as "important" (binary after thresholding).

    This means that the chosen model should be capable to produce input-attributional explanations. This
    kind of explanation assigns one or multiple "importance" value between 0 and 1 to each element of the
    input graph (nodes & edges) which determine how important that respective element was for the outcome of
    the target value prediction.
    """
    model_name = model

    # We are doing all the kgcnn imports only now because importing kgcnn will start the tensorflow runtime
    # which may take a few seconds and which may print a few error messages. In principle that is not a bad
    # thing in general, but when invoking the script with the "--help" option we do not want either to
    # happen. We want only the help text to be printed as quickly as possible without any overhead
    from kgcnn.data.visual_graph import VisualGraphDataset
    from kgcnn.hyper.hyper import HyperParameter
    from kgcnn.data.serial import deserialize as deserialize_dataset
    from kgcnn.utils.cli import echo_info, echo_success, echo_error
    from kgcnn.utils.models import get_model_class
    from kgcnn.utils.plots import plot_train_test_loss
    from kgcnn.training.history import save_history_score
    from kgcnn.xai.utils import flatten_importances_list

    # Tensorflow warnings are really annoying, so we only show them if the flag is explicitly set
    if not show_warnings:
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    echo_info(f'attempting training of model "{model}" and XAI method "{xai_method}" on dataset "{dataset}"')

    # == LOADING HYPER PARAMETERS ==
    # Technically the values provided through the command line options do not provide enough information to
    # accurately to fully specify a model training process. This is why the finer details should be encoded
    # within a dictionary structure inside an additional hyperparameter module, whose path we also need to
    # provide.
    
    # Given the path, this object will read that file and provide the correct configuration dict with all
    # the details about the training specification, given the model name and the dataset name (because
    # the hyperparameter module may contain specifications for different scenarios)
    echo_info(f'loading hyper parameters @ "{hyper}"')
    hyper_params = HyperParameter(
        hyper_info=hyper,
        model_name=model,
        model_class=make,
        dataset_name=dataset
    )

    # == CREATING RESULTS FOLDER ==
    # Since we are about to generate a bunch of artifacts, we create a new directory here where we are going
    # to save all of those artifacts into, so we don't accidentally clutter an important folder of the user.
    results_path = hyper_params.results_file_path()
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.mkdir(results_path)
    
    # == LOADING DATASET ==
    # hyper_params["data"]["dataset"] is a dictionary containing the detailed specification of the dataset
    # to be used for this experiment. "deserialize_dataset" will use the information from this dict to
    # create and return the corresponding dataset object.
    echo_info(f'loading dataset with name "{dataset}"')
    visual_graph_dataset: VisualGraphDataset = deserialize_dataset(hyper_params['data']['dataset'])

    if not isinstance(visual_graph_dataset, VisualGraphDataset):
        echo_error(f'the given dataset {dataset} is not a visual graph dataset and cannot be used with '
                   f'this command.')
        return 1

    # This method will make sure that the dataset exists in the filesystem, if it does not already it is
    # being downloaded
    visual_graph_dataset.ensure()
    echo_success(f'dataset located @ {visual_graph_dataset.data_directory}')

    # This method will read the actual dataset into memory
    visual_graph_dataset.read_in_memory()
    data_length = len(visual_graph_dataset)
    echo_success(f'loaded dataset with {data_length} elements into memory')

    # Model Selection to load a model definition from a module in kgcnn.literature
    num_importance_channels = int(hyper_params['explanation']['channels'])
    make_model = get_model_class(module_name=model, class_name=make)

    targets = np.array(visual_graph_dataset.obtain_property("graph_labels"))
    # TODO: Implement this for VGD
    #label_names = dataset.label_names
    #label_units = dataset.label_units

    # Cross-validation via random KFold split from `sklearn.model_selection`.
    # Or from dataset information.
    if hyper_params["training"]["cross_validation"] is None:
        echo_info("using dataset splits")
        # TODO: Add support for multiple splits
        train_test_indices = dataset.get_train_test_indices()

    else:
        kf = KFold(**hyper_params["training"]["cross_validation"]["config"])
        train_test_indices = [
            [train_index, test_index] for train_index, test_index in
            kf.split(X=np.zeros((data_length, 1)), y=targets)]

    # == MODEL TRAINING ===
    splits_done = 0
    history_list, test_indices_list = [], []
    model, hist, x_test, y_test, scaler, atoms_test = None, None, None, None, None, None
    for i, (train_indices, test_indices) in enumerate(train_test_indices):

        echo_info("Running training on fold: %s" % i)

        # Make the model for current split using model kwargs from hyperparameter.
        # They are always updated on top of the models default kwargs.
        model = make_model(**hyper_params["model"]["config"])

        # First select training and test graphs from indices, then convert them into tensorflow tensor
        # representation. Which property of the dataset and whether the tensor will be ragged is retrieved
        # from the kwargs of the keras `Input` layers ('name' and 'ragged').
        x_train = visual_graph_dataset[train_indices].tensor(hyper_params["model"]["config"]["inputs"])
        y_train = targets[train_indices]

        x_test = visual_graph_dataset[test_indices].tensor(hyper_params["model"]["config"]["inputs"])
        y_test = targets[test_indices]

        # Compile model with optimizer and loss
        metrics = None  # TODO: Handle metrics
        model.compile(**hyper_params.compile(loss="mean_absolute_error", metrics=metrics))
        print(model.summary())

        # Start and time training
        start_time = time.process_time()
        hist = model.fit(x_train, y_train,
                         validation_data=(x_test, y_test),
                         **hyper_params.fit())
        stop_time = time.process_time()
        echo_info(f"time for training: {stop_time - start_time:.1f} seconds")

        # Get loss from history
        history_list.append(hist)
        test_indices_list.append([train_indices, test_indices])

        splits_done = splits_done + 1
        echo_info(f'done processing split {splits_done}')

        # == CREATING EXPLANATIONS ==
        #
        node_importances, edge_importances = model.explain_importances(x_test)
        node_importances = [minmax_scale(a) for a in node_importances.numpy()]
        edge_importances = [minmax_scale(a) for a in edge_importances.numpy()]

        # ~ Explanation Accuracy
        # One metric which we can't always determine is the explanation accuracy because for that the
        # dataset needs some definite ground truth explanations, which is almost always only the case for
        # synthetic datasets.
        # This is why we first need to check the dataset whether it actually contains some gt
        # explanations.

        gt_importances_suffix = str(num_importance_channels)
        if hyper_params['explanation']['gt_suffix']:
            gt_importances_suffix = hyper_params['explanation']['gt_suffix']

        if visual_graph_dataset.has_importances(gt_importances_suffix):
            node_importances_gt, edge_importances_gt = visual_graph_dataset.get_importances(
                suffix=gt_importances_suffix,
                indices=test_indices,
            )
            # Now we can calculate how well the predicted explanations match the gt explanations using
            # the AUC metric (since all importance values are binary values between 0 and 1).
            # But currently the importances are still two-dimensional arrays. To calculate the AUC these
            # have to be flattened into one dimensional lists first.
            node_auc = roc_auc_score(
                flatten_importances_list(node_importances_gt),
                flatten_importances_list(node_importances)
            )
            edge_auc = roc_auc_score(
                flatten_importances_list(edge_importances_gt),
                flatten_importances_list(edge_importances)
            )
            echo_info(f'Node AUC: {node_auc} - Edge AUC: {edge_auc}')
            hist.history['node_auc'] = [0]
            hist.history['edge_auc'] = [0]
            hist.history['val_node_auc'] = [node_auc]
            hist.history['val_edge_auc'] = [edge_auc]

        # == CREATING VISUALIZATIONS ==
        example_indices = random.sample(
            test_indices.tolist(), 
            k=int(len(test_indices) * visualization_ratio)
        )
        example_indices = sorted(example_indices)
        echo_info(f'creating explanation visualizations for {len(example_indices)} indices from test set')

        # Now to create the visualizations for these chosen example elements, we first need to create the
        # explanations for them.
        x_example = visual_graph_dataset[example_indices].tensor([
            {'name': 'node_attributes', 'ragged': True},
            {'name': 'edge_attributes', 'ragged': True},
            {'name': 'edge_indices', 'ragged': True}
        ])
        node_importances, edge_importances = model.explain_importances(x_example)
        node_importances = node_importances.numpy()
        edge_importances = edge_importances.numpy()

        pdf_path = os.path.join(results_path, f'importances_split_{splits_done}.pdf')
        visual_graph_dataset.visualize_importances(
            output_path=pdf_path,
            gt_importances_suffix=str(num_importance_channels),
            node_importances_list=node_importances,
            edge_importances_list=edge_importances,
            indices=example_indices,
        )

    # == KGCNN RESULTS PROCESSING ==
    echo_info('plotting the loss over training epochs')
    postfix_file = hyper_params["info"]["postfix_file"]
    plot_train_test_loss(
        history_list,
        loss_name=None,
        val_loss_name=None,
        model_name=model_name,
        data_unit='',
        dataset_name=dataset,
        filepath=results_path,
        file_name=f'loss{postfix_file}.png'
    )

    echo_info('saving the training history')
    save_history_score(
        history_list,
        loss_name=None,
        val_loss_name=None,
        model_name=model_name,
        data_unit='',
        dataset_name=dataset,
        model_class=make,
        filepath=results_path,
        file_name=f'score{postfix_file}.yaml'
    )

    echo_info('saving hyper parameters')
    hyper_params.save(os.path.join(results_path, f'{model_name}_hyper{postfix_file}.json'))

    echo_success(f'View the results @ {results_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
