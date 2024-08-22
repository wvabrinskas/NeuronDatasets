# NeuronDatasets

Package contains machine learning datasets that are supported the [Neuron](https://github.com/wvabrinskas/Neuron) package. 

## Supported Datasets
| Dataset       | Origin
| ------------- | ------------- |
| CIFAR-10 | Local |
| MNIST    | Local |
| QuickDraw | Remote |

## Importing image datasets
You can import images from a directory to create a Dataset that Neuron can use. Useful for datasets you can download from Kaggle. Mostly useful for GAN and other generative networks.

Create an `ImageDataset` object

```
  let dataset = ImageDataset(imagesDirectory: URL(string: "/Users/williamvabrinskas/Desktop/ImageDataset")!,
                             imageSize: CGSize(width: 64, height: 64),
                             label: [1.0],
                             imageDepth: .rgb,
                             maxCount: 10000)
```
- `imagesDirectory`: The directory of the images to load. All images should be the same size.
- `imageSize`: The expected size of the images
- `label`: The label to apply to every image.
- `imageDepth`: ImageDepth that describes the expected depth of the images.
- `maxCount`: Max count to add to the dataset. Could be useful to save memory. Setting it to 0 will use the whole dataset.
- `validationSplitPercent`: Number between 0 and 1. The lower the number the more likely it is the image will be added to the training dataset otherwise it'll be added to the validation dataset.
- `zeroCentered`: Format image RGB values between -1 and 1. Otherwise it'll be normalized to between 0 and 1.

To build the dataset just call `.build()` on the dataset object.

## Importing a CSV dataset
You can import a CSV dataset from a directory to create a Dataset that Neuron can use using `CSVDataset`. 

```
// specifies the headers in the CSV files
enum TestHeaders: String, CSVSupporting {
  case id = "Id"
  case name = "Name"
  
  func order() -> [TestHeaders] {
    Self.allCases
  }
  
  func maxLengthOfItem() -> Int {
    switch self {
    case .name:
      return 10
    default:
      return 1
    }
  }
}

let path = Bundle.module.path(forResource: "smallBabyNamesTest", ofType: "csv") // test csv provided in the bundle

guard let path, let pathUrl = URL(string: path) else { return }

let splitPercentage: Tensor.Scalar = 0.2

let csvDataset = CSVDataset<TestHeaders>.init(csvUrl: pathUrl,
                                              headerToFetch: .name,
                                              validationSplitPercentage: splitPercentage,
                                              parameters: .init(oneHot: true))

let build = await csvDataset.build()

```

- `csvUrl`: the url of the CSV file
- `headerToFetch`: the K: Header value you want to fetch
- `maxCount`: the max number of objects you want. 0 = unlimited
- `validationSplitPercentage`: The validation split percentage to generate. min: 0.1, max: 0.9
- `overrideLabel`: the label to apply to each object. Otherwise the label will be set to the data. eg. `data: [0,1,0], label: [0,1,0]`
- `parameters`: The configuration parameters
## Utilities
In the `bin` folder there are some helpful scripts to help format image databases. 
| Script | Description | Usage |
| ------ | ----------- | ----- | 
| resize.py | will automatically resize images in a given directory to a specified size | `python3 ./bin/resize.py --width 64 --height 64 --path PATH_TO_IMAGES_DIR` |
