import argparse
import json
import logging
import os
import sys
#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from timeit import default_timer as timer
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, sampler
import numpy as np
import pandas as pd
import shlex, subprocess
from torchsummary import summary
# def install(package):
#     os.system("pip install " +  package)
    
# install('pillow')
# install('requests')
# install('pydicom')
from PIL import Image
import requests
import pydicom

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2,
          train_on_gpu=False):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except Exception:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = (
                np.squeeze(correct_tensor.cpu().numpy())
                if train_on_gpu
                else np.squeeze(correct_tensor.numpy())
            )
            # calculate test accuracy for each object class
            '''for i in range(batch_size):       
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1'''

            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        model.epochs += 1

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            for data, target in valid_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Forward pass
                output = model(data)

                # Validation loss
                loss = criterion(output, target)
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)

                # Calculate validation accuracy
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(
                    correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * data.size(0)

            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            # Calculate average accuracy
            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(valid_loader.dataset)

            history.append([train_loss, valid_loss, train_acc, valid_acc])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                    )
                    total_time = timer() - overall_start
                    print(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )

                    # Load the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_loss', 'valid_loss', 'train_acc',
                            'valid_acc'
                        ])
                    return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
#     print(
#         f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
#     )
#     print(
#         f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
#     )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


def save_checkpoint(model, path, multi_gpu):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if model_name == 'vgg16':
        # Check to see if model was parallelized
        if multi_gpu:
            checkpoint['classifier'] = model.module.classifier
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'resnet50':
        if multi_gpu:
            checkpoint['fc'] = model.module.fc
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['fc'] = model.fc
            checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


    


def create_dataloader(traindir,testdir,validdir,batch_size=128):
    image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ])
        ,
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    }
    
    
    data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }

    
    return data,dataloaders

def create_models():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    return model
        
def preops(traindir,testdir,validdir):
    print("********* inside preops *************")
    print(f"{traindir} , {testdir} , {validdir}")
    # Empty lists
    categories = []
    img_categories = []
    n_train = []
    n_valid = []
    n_test = []
    hs = []
    ws = []

    # Iterate through each category
    for d in os.listdir(traindir):
        if not d.startswith('.'):
            categories.append(d)

            # Number of each image
            train_imgs = os.listdir(traindir + d)
            valid_imgs = os.listdir(validdir + d)
            test_imgs = os.listdir(testdir + d)
            n_train.append(len(train_imgs))
            n_valid.append(len(valid_imgs))
            n_test.append(len(test_imgs))

            # Find stats for train images
            for i in train_imgs:
                if not i.startswith('.'):
                    img_categories.append(d)
                    img = Image.open(traindir + d + '/' + i).convert("RGB")
                    img_array = np.array(img)
                    # Shape
                    hs.append(img_array.shape[0])
                    ws.append(img_array.shape[1])

    # Dataframe of categories
    cat_df = pd.DataFrame({'category': categories,
                           'n_train': n_train,
                           'n_valid': n_valid, 'n_test': n_test}).\
        sort_values('category')

    # Dataframe of training images
    image_df = pd.DataFrame({
        'category': img_categories,
        'height': hs,
        'width': ws
    })
    
    return cat_df

def get_pretrained_model(model_name,n_classes):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    return model

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model, path)

def load_checkpoint(path,multi_gpu):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Get the model name
    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Load in checkpoint
    checkpoint = torch.load(path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    # if train_on_gpu:
    #     model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer
# model, optimizer = load_checkpoint(path=checkpoint_path)

# if multi_gpu:
#     summary(model.module, input_size=(3, 224, 224), batch_size=batch_size)
# else:
#     summary(model, input_size=(3, 224, 224), batch_size=batch_size)


'''
inferencing
'''

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(Net())
    # with open(os.path.join(model_dir, "model.pth"), "rb") as f:
    #     model.load_state_dict(torch.load(f))
    f = "model.pth"
    model = torch.load(os.path.join(model_dir, "model.pth"))
    model.eval()
    return model.to(device)


def input_fn(request_body, content_type='application/json'):
    # sourcery skip: raise-specific-error
    logger.info('Deserializing the input data.')
    import requests
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if content_type == 'application/json':
        # request_body = json.dumps({"url":"http://rasbt.github.io/mlxtend/user_guide/data/mnist_data_files/mnist_data_10_0.png"})    
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # resp = requests.get(url, stream=True)
        # raw_data = resp.raw
        image_data = Image.open(requests.get(url, stream=True).raw).convert('RGB')

        image_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # logger.info(image_data)
        data = image_transform(image_data)
        logger.info(str(data.shape))
        data = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
        logger.info(str(data.shape))
        return data
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def predict_fn(input_object, model):
    with torch.no_grad():
        logger.info(str(input_object.shape))
        logger.info(model.classifier)
        prediction = model(input_object)

    return prediction

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    finding = torch.exp(torch.tensor(predictions))
    res = finding.numpy().tolist()[0][1] * 100
    output = response_converter(res)
    return json.dumps(output)


def response_converter(res):
    '''
    THIS IS A CUSTOM RESPONSE CCONVERTER
    CARPL EXPECTS OUTPUT IN THIS FORMAT:
    {
        "response": {
            "findings":[
                {
                    "name":"class_A",
                    "probability":score_A
                },
                {
                    "name":"class_B",
                    "probability":score_B
                }
            ],
            "rois":[
                    {
                        "finding_name":"class1",
                        "type":"Rectangle",
                        "points":[
                            [50,300],
                            [100,200]
                        ]
                    },
                    {
                        "finding_name":"class2",
                        "type":"Freehand",
                        "points":[
                            [500,300],
                            [500,320],
                            [500,420],
                            [800,420],
                            [800,370],
                            [800,320]
                        ]
                    }
                ]
        }
    }
    '''
    import numpy as np
    try:

        result = []
        # for e,val in enumerate(res[0]):
        result.append(
            {
                "name":"class",
                "probability":int(res)
            }
        )

        return {
            "response" : {
                "findings":result,
                "rois":[
                    {
                        "finding_name":"class1",
                        "type":"Rectangle",
                        "points":[
                            [1050,1300],
                            [1100,1200]
                        ]
                    },
                    {
                        "finding_name":"class2",
                        "type":"Freehand",
                        "points":[
                            [900,1300],
                            [900,1320],
                            [900,1420],
                            [1300,1420],
                            [1300,1370],
                            [1300,1320]
                        ]
                    }
                ]
            }
        }
    except Exception as e:
        print(e) 



def run(args):

    is_distributed = len(args.hosts) > 1 and args.backend is not None

    # datadir = '/opt/ml/xray-nano/nano'
    traindir = args.data_dir_train + "/" #datadir + '/train/'
    validdir = args.data_dir_val + "/" #datadir + '/val/'
    testdir = args.data_dir_test + "/" #datadir + '/test/'
    # print(os.listdir(datadir))
    print("********* inside run *************")
    print(f"{traindir} , {testdir} , {validdir}")

    save_file_name = 'vgg16-chest-4.pt'
    checkpoint_path = 'vgg16-chest-4.pth'

    # Change to fit hardware
    batch_size = 128
    # Whether to train on a gpu

    use_cuda = args.num_gpus > 0
    logger.debug(f"Number of gpus available - {args.num_gpus}")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    # train_on_gpu = cuda.is_available()
    # print(f'Train on gpu: {train_on_gpu}')

    train_on_gpu = use_cuda

    # # Number of gpus
    # if train_on_gpu:
    #     gpu_count = cuda.device_count()
    #     print(f'{gpu_count} gpus detected.')
    #     if gpu_count > 1:
    #         multi_gpu = True
    #     else:
    #         multi_gpu = False

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    cat_df = preops(traindir,testdir,validdir)

    data,dataloaders = create_dataloader(traindir,testdir,validdir,batch_size)

    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
    print(features.shape, labels.shape)

    n_classes = len(cat_df)
    print(f'There are {n_classes} different classes.')

    print(len(data['train'].classes))
    #train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    #test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    model = create_models()
    n_inputs = model.classifier[6].in_features

    # Add on classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    model.classifier
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # if train_on_gpu:
    #     model = model.to('cuda')

    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    model = get_pretrained_model('vgg16',n_classes,)
    if use_cuda:
        summary(
            model.module,
            input_size=(3, 224, 224),
            batch_size=batch_size,
            device='cuda')
    else:
        summary(
            model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')

    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }


    print(list(model.idx_to_class.items()))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    for p in optimizer.param_groups[0]['params']:
        if p.requires_grad:
            print(p.shape)


    model, history = train(
        model,
        criterion,
        optimizer,
        dataloaders['train'],
        dataloaders['val'],
        save_file_name=save_file_name,
        max_epochs_stop=5,
        n_epochs=1,
        print_every=2,
        train_on_gpu=train_on_gpu)

    print(model,history)


    save_file_name = 'vgg16-chest-4.pt'
    checkpoint_path = 'vgg16-chest-4.pth'
    # save_checkpoint(model, checkpoint_path,use_cuda)
    save_model(model, args.model_dir)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir-train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--data-dir-test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--data-dir-val", type=str, default=os.environ["SM_CHANNEL_VALIDATING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    

    run(parser.parse_args())
