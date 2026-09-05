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

## Importing a text sequence dataset
You can import a column of text from a CSV file to create a next-token dataset that Neuron's `RNN` can use, using `TextSequenceDataset`. The dataset trains a `BPETokenizer` on the column it reads, then emits `(data, label)` pairs where the label is the input advanced by one token, so timestep `i` is scored on the token that follows it.

```
// specifies the headers in the CSV files
enum TestHeaders: String, TextSequenceDatasetSupporting {
  case id = "Id"
  case name = "Name"
}

let path = Bundle.module.path(forResource: "smallBabyNamesTest", ofType: "csv") // test csv provided in the bundle

guard let path, let pathUrl = URL(string: path) else { return }

let splitPercentage: Tensor.Scalar = 0.2

let dataset = TextSequenceDataset<TestHeaders>(csvUrl: pathUrl,
                                               headerToFetch: .name,
                                               targetVocabSize: 25,
                                               validationSplitPercentage: splitPercentage)

let build = await dataset.build()
```

- `csvUrl`: the url of the CSV file
- `headerToFetch`: the K: Header value you want to fetch
- `targetVocabSize`: the vocabulary size the `BPETokenizer` will merge towards while training on the column
- `maxCount`: the max number of objects you want. 0 = unlimited
- `validationSplitPercentage`: The validation split percentage to generate. min: 0.1, max: 0.9
- `overrideLabel`: the label to apply to each object. Otherwise the label is the input shifted forward by one token.
- `filter`: an optional `CharacterSet` trimmed from the ends of each row's value.

Your header `enum` only needs to conform to `TextSequenceDatasetSupporting`, which is a `String` `RawRepresentable` that is `CaseIterable`. The raw values must match the column names in the CSV.

### Working with tokens
`TextSequenceDataset` inherits from `ProxyTokenizableDataset`, so the tokenizer it trained is available on the dataset itself:

```
dataset.vocabSize            // size of the trained vocabulary
dataset.bosTokenId           // beginning of sequence control token
dataset.eosTokenId           // end of sequence control token
dataset.padTokenId           // padding control token
dataset.controlTokenIds      // all of the above

dataset.item(for: sample.data)                        // decode a Tensor back into a String
dataset.tokenize("Mary", paddedTo: 10, appendingEnd: true) // encode a String into a Tensor
dataset.tokenCount(for: "Mary")
dataset.sequenceLength(for: ["Mary", "John"], cappedAt: nil)
```

Decoding skips the control tokens, so a sample and its label render as the same text — the one token shift between them is only visible at the token level.

A trained tokenizer can be exported and reloaded so you don't have to retrain it on every run:

```
let url = dataset.export(name: "names-tokenizer", overrite: true, compress: true)

let restored = ProxyTokenizableDataset.build(url: url!) // also available as build(data:)
```

## Utilities
In the `bin` folder there are some helpful scripts to help format image databases. 
| Script | Description | Usage |
| ------ | ----------- | ----- | 
| resize.py | will automatically resize images in a given directory to a specified size | `python3 ./bin/resize.py --width 64 --height 64 --path PATH_TO_IMAGES_DIR` |
