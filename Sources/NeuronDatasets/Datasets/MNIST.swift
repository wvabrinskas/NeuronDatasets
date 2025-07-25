import Foundation
import NumSwift
import Combine
import Neuron
 
/// Creates an MNIST Dataset object to be used by a netowrk. The MNIST dataset is a set of 60000 grayscale hand-drawn numbers from 0-9.
public class MNIST: BaseDataset, DatasetMergable {
  private let mnistSize: [Int] = [28,28,1]
  private var numToGet: Int?
  private var zeroCentered: Bool

  public enum MNISTType: String, CaseIterable {
    case trainingSet = "train-images"
    case trainingLabels = "train-labels"
    case valSet = "t10k-images"
    case valLabels = "t10k-labels"
    
    var startingByte: UInt8 {
      switch self {
      case .trainingSet:
        return 0x0013
      case .trainingLabels:
        return 0x0008
      case .valSet:
        return 0x0013
      case .valLabels:
        return 0x0008
      }
    }
    
    var shape: [Int] {
      switch self {
      case .trainingSet, .valSet:
        return [28,28,1]
      case .trainingLabels, .valLabels:
        return [1,1,1]
      }
    }
    
    var modifier: Tensor.Scalar {
      switch self {
      case .trainingSet, .valSet:
        return 255.0
      case .valLabels, .trainingLabels:
        return 1.0
      }
    }
  }
  
  /// Default initializer for MNIST
  /// - Parameters:
  ///   - num: Optional parameter to only get a specific number from the Dataset.
  ///   - label: The label override for each image. Optional as the MNIST dataset provides labels.
  ///   - zeroCentered: Determines if the dataset is scaled to be between -1 and 1 or between 0 and 1.
  public init(only num: Int? = nil,
              overrideLabel: [Tensor.Scalar] = [],
              zeroCentered: Bool = false) {
    if let num = num {
      self.numToGet = num
    }

    self.zeroCentered = zeroCentered
    
    super.init(unitDataSize: .init(rows: 28, columns: 28, depth: 1),
               overrideLabel: overrideLabel)
  }
  
  /// Build the actual dataset using the async/await system
  /// - Parameter num: Optional parameter to only get a specific number from the Dataset.
  /// - Returns: The MNIST dataset object
  public func build(only num: Int) async -> DatasetData {
    self.numToGet = num
    return await build()
  }
  
  public func merge(with dataset: MNIST) {
    super.merge(with: dataset)
  }
  
  /// Build with support for Combine
  public override func build() {
    Task {
      await build()
    }
  }
  
  /// Build with async/await support
  /// - Returns: downloaded dataset
  @discardableResult
  public override func build() async -> DatasetData {
    guard complete == false else {
      print("MNIST has already been loaded")
      return self.data
    }
    
    print("Loading MNIST dataset into memory. This could take a while")
    
    
    self.data = await withTaskGroup(of: (data: [[[Tensor.Scalar]]], type: MNISTType).self, body: { group in
      
      var trainingDataSets: [(data: [[[Tensor.Scalar]]], type: MNISTType)] = []
      trainingDataSets.reserveCapacity(2)
      
      var valDataSets: [(data: [[[Tensor.Scalar]]], type: MNISTType)] = []
      valDataSets.reserveCapacity(2)
      
      group.addTask(priority: .userInitiated) {
        let trainingData = self.get(type: .trainingSet)
        return (trainingData, .trainingSet)
      }
      
      group.addTask(priority: .userInitiated) {
        let trainingData = self.get(type: .trainingLabels)
        return (trainingData, .trainingLabels)
      }
      
      group.addTask(priority: .userInitiated) {
        let trainingData = self.get(type: .valSet)
        return (trainingData, .valSet)
      }
      
      group.addTask(priority: .userInitiated) {
        let trainingData = self.get(type: .valLabels)
        return (trainingData, .valLabels)
      }
      
      for await data in group {
        if data.type == .trainingLabels || data.type == .trainingSet {
          trainingDataSets.append(data)
        } else if data.type == .valLabels || data.type == .valSet {
          valDataSets.append(data)
        }
      }
      
      var trainingDataWithLabels: [DatasetModel] = []
      var validationDataWithLabels: [DatasetModel] = []
      
      let validationData = valDataSets.first { $0.type == .valSet }?.data ?? []
      let validationLabels = valDataSets.first { $0.type == .valLabels }?.data ?? []
      
      let trainingData = trainingDataSets.first { $0.type == .trainingSet }?.data ?? []
      let trainingLabels = trainingDataSets.first { $0.type == .trainingLabels }?.data ?? []
      
      for i in 0..<trainingData.count {
        let tD = trainingData[i]
        let tL = trainingLabels[i].first?.first ?? -1
        if numToGet == nil || Int(tL) == numToGet {
          let conv = DatasetModel(data: Tensor(tD), label: Tensor(self.buildLabel(value: Int(tL)))) //only one channel for MNIST
          trainingDataWithLabels.append(conv)
        }
      }
      
      for i in 0..<validationData.count {
        let tD = validationData[i]
        let tL = validationLabels[i].first?.first ?? -1
        if numToGet == nil || Int(tL) == numToGet {
          let conv = DatasetModel(data: Tensor(tD), label: Tensor(self.buildLabel(value: Int(tL)))) //only one channel for MNIST
          validationDataWithLabels.append(conv)
        }
      }
      
      return (trainingDataWithLabels, validationDataWithLabels)
    })
    
    return await super.build()
  }
  
  private func buildLabel(value: Int) -> [Tensor.Scalar] {
    if !self.overrideLabel.isEmpty {
      return overrideLabel
    }
    
    guard value >= 0 else {
      return []
    }
    var labels = [Tensor.Scalar].init(repeating: 0, count: 10)
    labels[value] = 1
    return labels
  }
  
  public func get(type: MNISTType) -> [[[Tensor.Scalar]]] {
    let path = Bundle.module.path(forResource: type.rawValue, ofType: nil)
    
    guard let path = path else {
      return []
    }
    
    let shouldZeroCenter: Bool = zeroCentered && (type == .valSet || type == .trainingSet)
    
    var scale: Tensor.Scalar = shouldZeroCenter ? 1 : 255
    
    if type == .trainingLabels || type == .valLabels {
      scale = type.modifier
    }
    
    var result = read(path: path, offset: Int(type.startingByte), scaleBy: scale)
    
    if shouldZeroCenter {
      result = result.map { ($0 - 127.5) / 127.5 }
    }
    
    let columns = type.shape[safe: 0] ?? 0
    let rows = type.shape[safe: 1] ?? 0
    
    let shaped: [[[Tensor.Scalar]]] = result.reshape(columns: columns).batched(into: rows).compactMap { value in
      guard value.shape == [rows, columns] else {
        log(type: .error, message: "invalid shape in image: \(value.shape)")
        return nil
      }
      
      return value
    }
    
    return shaped
  }
}
