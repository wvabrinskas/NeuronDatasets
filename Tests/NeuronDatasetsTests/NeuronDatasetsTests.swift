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
  
  func test_trim() async {
    let trim = 10
    let mnist = MNIST()
    mnist.trim(to: trim)
    let result = await mnist.build()
    
    XCTAssertEqual(result.training.count, trim)
    XCTAssertEqual(result.val.count, trim)
    
    let cfar = CIFAR(classType: .airplane)
    cfar.trim(to: trim)
    let c = await cfar.build()
    
    XCTAssertEqual(c.training.count, trim)
    // CIFAR doesn't have validation
  }
  
  func testCSVDataset_unvectorize() async {
    enum TestHeaders: String, CSVSupporting {
      case id = "Id"
      case name = "Name"
      
      var type: CSVType { .character }
      
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
      
      var type: CSVType { .character }
      
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
  
  func testImageDatasetLabelsCheck() {
    ImageDataset.ImageDepth.allCases.forEach { depth in
      let imageSize = CGSize(width: 20, height: 20)
      let imageLabels = URL(string: Bundle.module.path(forResource: "test-image-labels", ofType: "csv")!)
    
      let dataset = ImageDataset(imagesDirectory: URL(string: "https://images.com")!,
                                 labels: imageLabels,
                                 imageSize: imageSize,
                                 label: [1.0],
                                 imageDepth: depth)
      
      do {
        let labels = try dataset.getLabelsIfNeeded()
        let expectedLabels: [[Tensor.Scalar]] = [[1,0,0,0,0],
                                         [1,0,0,0,0],
                                         [1,0,0,0,0],
                                         [1,0,0,0,0],
                                         [0,1,0,0,0],
                                         [0,1,0,0,0],
                                         [0,1,0,0,0],
                                         [0,1,0,0,0],
                                         [0,0,1,0,0],
                                         [0,0,1,0,0],
                                         [0,0,1,0,0],
                                         [0,0,1,0,0],
                                         [0,0,0,1,0],
                                         [0,0,0,1,0],
                                         [0,0,0,1,0],
                                         [0,0,0,1,0],
                                         [0,0,0,0,1],
                                         [0,0,0,0,1],
                                         [0,0,0,0,1],
                                         [0,0,0,0,1]]
        XCTAssertEqual(labels?.count, 4 * 5)
        let flat = labels!.map { $0.value }.map { $0.flatten() }
        XCTAssertEqual(flat, expectedLabels)
      } catch {
        print(error.localizedDescription)
      }
 
      
      XCTAssertEqual(dataset.unitDataSize, TensorSize(rows: Int(imageSize.height),
                                                      columns: Int(imageSize.width),
                                                      depth: depth.expectedDepth))
    }
  }
  
  func testCSVDataset_Sentence() async {
    enum TestHeaders: String, CSVSupporting {
      case username = "user_name"
      case userLocation = "user_location"
      case userDescription = "user_description"
      case userCreated = "user_created"
      case userFollowers = "user_followers"
      case userFriends = "user_friends"
      case userFavourites = "user_favourites"
      case userVerified = "user_verified"
      case date
      case text
      case hashtags
      case source
      case isRetweet = "isRetweet"
      //user_name,user_location,user_description,user_created,user_followers,user_friends,user_favourites,user_verified,date,text,hashtags,source,is_retweet
      
      var type: CSVType { .sentence }
      
      func order() -> [TestHeaders] {
        Self.allCases
      }
      
      func maxLengthOfItem() -> Int {
        switch self {
        case .text:
          return 140
        default:
          return 1
        }
      }
    }
    
    let path = Bundle.module.path(forResource: "sentenceTweetsSmallTest", ofType: "csv")
    
    XCTAssertNotNil(path)
    guard let path, let pathUrl = URL(string: path) else { return }
    
    let splitPercentage: Float = 0.2
    
    let csvDataset = CSVDataset<TestHeaders>.init(csvUrl: pathUrl,
                                                  headerToFetch: .text,
                                                  validationSplitPercentage: splitPercentage,
                                                  parameters: .init(oneHot: true))
    
    let build = await csvDataset.build()
    
    let name = build.training[0].data // should be "mary" with depth of 1
    
    let unvectorized = csvDataset.getWord(for: name).filter { $0 != "." }.joined()
    
    XCTAssertEqual(unvectorized, "Which #bitcoin books should I think about reading next? https://t.co/32gas26rKB".lowercased())
    
    let label = build.training[0].label
    
    let unvectorizedLabel = csvDataset.getWord(for: label).filter { $0 != "." }.joined()
    
    XCTAssertEqual(unvectorizedLabel, " #bitcoin books should i think about reading next? https://t.co/32gas26rkb".lowercased())
  }
  
  
  func testLSTM() async {
    
    guard isGithubCI == false else {
      XCTAssert(true)
      return
    }
    
//    enum TestHeaders: String, CSVSupporting {
//      case username = "user_name"
//      case userLocation = "user_location"
//      case userDescription = "user_description"
//      case userCreated = "user_created"
//      case userFollowers = "user_followers"
//      case userFriends = "user_friends"
//      case userFavourites = "user_favourites"
//      case userVerified = "user_verified"
//      case date
//      case text
//      case hashtags
//      case source
//      case isRetweet = "isRetweet"
//      //user_name,user_location,user_description,user_created,user_followers,user_friends,user_favourites,user_verified,date,text,hashtags,source,is_retweet
//      
//      var type: CSVType { .sentence }
//      
//      func order() -> [TestHeaders] {
//        Self.allCases
//      }
//      
//      func maxLengthOfItem() -> Int {
//        switch self {
//        case .text:
//          return 5 // word count
//        default:
//          return 1
//        }
//      }
//    }
//    
//    let path = Bundle.module.path(forResource: "sentenceTweetsSmallTest", ofType: "csv")
    
    enum TestHeaders: String, CSVSupporting {
      case id = "Id"
      case name = "Name"
      
      var type: CSVType { .character }
      
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
    
    let reporter = MetricsReporter(frequency: 1,
                                   metricsToGather: [.loss,
                                                     .accuracy,
                                                     .valAccuracy,
                                                     .valLoss])
    
    let rnn = RNN(returnSequence: true,
                  dataset: csvDataset,
                  classifierParameters: RNN.ClassifierParameters(batchSize: 64,
                                                                 epochs: 100,
                                                                 accuracyThreshold: 0.8,
                                                                 threadWorkers: 8),
                  optimizerParameters: RNN.OptimizerParameters(learningRate: 0.0002,
                                                               metricsReporter: reporter),
                  lstmParameters: RNN.RNNLSTMParameters(hiddenUnits: 256,
                                                        inputUnits: 100))// {
    //      [
    //       Dense(64),
    //       ReLu(),
    //       Dropout(0.5),
    //       Dense(vocabSize),
    //       Softmax()]
    //    }
    //
    reporter.receive = { metrics in
      let accuracy = metrics[.accuracy] ?? 0
      let loss = metrics[.loss] ?? 0
     // print("training -> ", "loss: ", loss, "accuracy: ", accuracy)
    }
    
    rnn.onEpochCompleted = {
      let word = rnn.predict(count: 10, randomizeSelection: true)
      print(word)
    }
    
    rnn.onAccuracyReached = {
      let word = rnn.predict(count: 10, randomizeSelection: true)
      print(word)
    }
    
    await rnn.train()
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
                                                     .valLoss,
                                                     .batchTime])
    
    optim.metricsReporter = reporter
    
    optim.metricsReporter?.receive = { metrics in
      let accuracy = metrics[.accuracy] ?? 0
      let loss = metrics[.loss] ?? 0
      print("batchTime: ", metrics[.batchTime] ?? 0)
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
