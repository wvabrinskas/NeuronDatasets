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

  func testMNISTClassifier() async {
    guard isGithubCI == false else {
      XCTAssert(true)
      return
    }
    
    let dataset = MNIST()
    let initializer: InitializerType = .heNormal
    
    let flatten = Flatten()
    flatten.inputSize = TensorSize(rows: dataset.unitDataSize.rows,
                                   columns: dataset.unitDataSize.columns,
                                   depth: dataset.unitDataSize.depth)
    
    let network = Sequential {
      [
        Conv2d(filterCount: 16,
               inputSize: flatten.inputSize,
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
    //optim.device = GPU()
    
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
                                batchSize: 16,
                                threadWorkers: 16,
                                log: false)
    
    let data = await dataset.build()
  
    classifier.fit(data.training, data.val)
  }
}
