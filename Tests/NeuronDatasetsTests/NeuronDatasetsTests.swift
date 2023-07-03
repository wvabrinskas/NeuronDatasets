import XCTest
import Neuron
@testable import NeuronDatasets

extension XCTestCase {
  var isGithubCI: Bool {
    if let value = ProcessInfo.processInfo.environment["CI"] {
      return value == "true"
    }
    return false
  }
}

final class NeuronDatasetsTests: XCTestCase {
  
  func testCSVDataset_unvectorize() async {
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
    
    let path = Bundle.module.path(forResource: "smallBabyNamesTest", ofType: "csv")
    
    XCTAssertNotNil(path)
    guard let path, let pathUrl = URL(string: path) else { return }
    
    let splitPercentage: Float = 0.2
    
    let csvDataset = CSVDataset<TestHeaders>.init(csvUrl: pathUrl,
                                                  headerToFetch: .name,
                                                  validationSplitPercentage: splitPercentage,
                                                  parameters: .init(oneHot: true))
    
    let build = await csvDataset.build()
    
    let name = build.training[0].data // should be "mary" with depth of 1
    
    let unvectorized = csvDataset.getWord(for: name).filter { $0 != "." }.joined()
    
    XCTAssertEqual(unvectorized, "mary")
    
    let label = build.training[0].label
    
    let unvectorizedLabel = csvDataset.getWord(for: label).filter { $0 != "." }.joined()
    
    XCTAssertEqual(unvectorizedLabel, "ary")
  }
  
  func testCSVDataset() async {
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
    
    let path = Bundle.module.path(forResource: "smallBabyNamesTest", ofType: "csv")
    
    XCTAssertNotNil(path)
    guard let path, let pathUrl = URL(string: path) else { return }
    
    let splitPercentage: Float = 0.2
    
    let csvDataset = CSVDataset<TestHeaders>.init(csvUrl: pathUrl,
                                                  headerToFetch: .name,
                                                  validationSplitPercentage: splitPercentage,
                                                  parameters: .init(oneHot: true))
    
    let build = await csvDataset.build()
    
    let trainingCount = Int(floor(Float(970 - 1) * Float(1 - splitPercentage)))
    let valCount = (970 - 1) - trainingCount

    XCTAssertEqual(build.training.count, trainingCount)
    XCTAssertEqual(build.val.count, valCount)
  }
  
  func testImageDatasetDepthCheck() {
    ImageDataset.ImageDepth.allCases.forEach { depth in
      let imageSize = CGSize(width: 20, height: 20)
      let dataset = ImageDataset(imagesDirectory: URL(string: "https://images.com")!,
                                 imageSize: imageSize,
                                 label: [1.0],
                                 imageDepth: depth)
      
      XCTAssertEqual(dataset.unitDataSize, TensorSize(rows: Int(imageSize.height),
                                                      columns: Int(imageSize.width),
                                                      depth: depth.expectedDepth))
    }
  }
  
  func testLSTM() async {
    
    guard isGithubCI == false else {
      XCTAssert(true)
      return
    }
    
    enum TestHeaders: String, CSVSupporting {
      case id = "Id"
      case name = "Name"
      
      func order() -> [TestHeaders] {
        Self.allCases
      }
      
      func maxLengthOfItem() -> Int {
        switch self {
        case .name:
          return 15
        default:
          return 1
        }
      }
    }
    
    let path = Bundle.module.path(forResource: "HamsterNames", ofType: "csv")
    
    XCTAssertNotNil(path)
    guard let path, let pathUrl = URL(string: path) else { return }
    
    let splitPercentage: Float = 0.1
    
    let csvDataset = CSVDataset<TestHeaders>.init(csvUrl: pathUrl,
                                                  headerToFetch: .name,
                                                  labelOffset: 0,
                                                  validationSplitPercentage: splitPercentage,
                                                  parameters: .init(oneHot: true),
                                                  filter: CharacterSet(charactersIn: "\r"))
    
    let reporter = MetricsReporter(frequency: 32,
                                   metricsToGather: [.loss,
                                                     .accuracy,
                                                     .valAccuracy,
                                                     .valLoss])
    
    let rnn = RNN(returnSequence: true,
                  dataset: csvDataset,
                  classifierParameters: RNN.ClassifierParameters(batchSize: 16,
                                                                 epochs: 400,
                                                                 accuracyThreshold: 0.8,
                                                                 threadWorkers: 8),
                  optimizerParameters: RNN.OptimizerParameters(learningRate: 0.001,
                                                               metricsReporter: reporter),
                  lstmParameters: RNN.RNNLSTMParameters(hiddenUnits: 64,
                                                        inputUnits: 50,
                                                        embeddingInitializer: .xavierNormal,
                                                        lstmInitializer: .xavierNormal))
    
    let predict = false
    let importTrain = false
//
    if predict || importTrain {
      if let importUrl = Bundle.module.path(forResource: "hamster-7", ofType: "smodel") {
        await rnn.importFrom(url: URL(string: "file://" + importUrl))
      }
    }

    reporter.receive = { metrics in
      let accuracy = metrics[.accuracy] ?? 0
      let loss = metrics[.loss] ?? 0
      print("training -> ", "loss: ", loss, "accuracy: ", accuracy)
    }
    
    rnn.onEpochCompleted = {
      print(rnn.export(overrite: true))
      let word = rnn.predict(count: 10, maxWordLength: 8, randomizeSelection: true)
      let nonRandomWord = rnn.predict(count: 10, maxWordLength: 8, randomizeSelection: false)

      print("word", word)
      print("nonrandom", nonRandomWord)
    }
    
    rnn.onAccuracyReached = {
      let word = rnn.predict(count: 10, maxWordLength: 8, randomizeSelection: true)
      print(word)
    }
    
    if predict {
      let word = rnn.predict(count: 20, maxWordLength: 8, randomizeSelection: false)
      print(word)
    } else {
      await rnn.train()
    }

  }

  func testMNISTClassifier() async {
    guard isGithubCI == false else {
      XCTAssert(true)
      return
    }
    
    let initializer: InitializerType = .heNormal
    
    let flatten = Flatten()
    flatten.inputSize = TensorSize(array: [28, 28, 1])
    
    let network = Sequential {
      [
        Conv2d(filterCount: 16,
               inputSize: TensorSize(array: [28,28,1]),
               padding: .same,
               initializer: initializer),
        BatchNormalize(),
        LeakyReLu(limit: 0.2),
        MaxPool(),
        Conv2d(filterCount: 32,
               padding: .same,
               initializer: initializer),
        BatchNormalize(),
        LeakyReLu(limit: 0.2),
        Dropout(0.5),
        MaxPool(),
        Flatten(),
        Dense(64, initializer: initializer),
        LeakyReLu(limit: 0.2),
        Dense(10, initializer: initializer),
        Softmax()
      ]
    }
    
    let optim = Adam(network, learningRate: 0.0001, l2Normalize: false)
    
    let reporter = MetricsReporter(frequency: 1,
                                   metricsToGather: [.loss,
                                                     .accuracy,
                                                     .valAccuracy,
                                                     .valLoss])
    
    optim.metricsReporter = reporter
    
    optim.metricsReporter?.receive = { metrics in
      let accuracy = metrics[.accuracy] ?? 0
      let loss = metrics[.loss] ?? 0
      print("training -> ", "loss: ", loss, "accuracy: ", accuracy)
    }
    
    let classifier = Classifier(optimizer: optim,
                                epochs: 10,
                                batchSize: 32,
                                threadWorkers: 8,
                                log: false)
    
    let data = await MNIST().build()
  
    classifier.fit(data.training, data.val)
  }
}
