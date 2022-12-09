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
      //let valLoss = metrics[.valLoss] ?? 0

      print("training -> ", "loss: ", loss, "accuracy: ", accuracy)
    }

    let classifier = Classifier(optimizer: optim,
                                epochs: 10,
                                batchSize: 32,
                                threadWorkers: 8,
                                log: false)

    let data = await MNIST().build()

    /*
     for _ in 0..<3 {
       if let random = data.val.randomElement() {
         let label = random.label
         let data = random.data

         let labelArray: [Float] = label.value.flatten()
         let labelMax = labelArray.indexOfMax.0

         let out = classifier.feed([data])
         if let first = out.first {
           let firstFlat: [Float] = first.value.flatten()
           let firstMax = firstFlat.indexOfMax.0

           print("out: ", firstMax, "- \(firstFlat.indexOfMax.1 * 100)%", "---> expected: ", labelMax, "- \(labelArray.indexOfMax.1 * 100)%")
           XCTAssert(firstMax == labelMax)
         }
       }
     }
     */
    classifier.fit(data.training, data.val)
  }
}
