//
//  File.swift
//  
//
//  Created by William Vabrinskas on 11/29/22.
//

import Foundation
import Combine
import Neuron

public final class CIFAR: Dataset {
  public enum ClassType: Int {
   case airplane,
    automobile,
    bird,
    cat,
    deer,
    dog,
    frog,
    horse,
    ship,
    truck
  }
  
  public var data: DatasetData = ([], []) {
    didSet {
      dataPassthroughSubject.send(data)
    }
  }
  
  public var complete: Bool = false
  public let unitDataSize: TensorSize = .init(rows: 32, columns: 32, depth: 3)
  
  public var dataPassthroughSubject = PassthroughSubject<DatasetData, Never>()
  public var overrideLabel: [Float] = []
  private let classType: ClassType
  
  public init(classType: ClassType, overrideLabel: [Float] = []) {
    self.overrideLabel = overrideLabel
    self.classType = classType
  }
  
  public func build() async -> DatasetData {
    guard complete == false else {
      print("CIFAR has already been loaded")
      return data
    }
    
    print("Loading CIFAR dataset into memory. This could take a while")
    
    complete = true
    
    let datasetData = get()
    data = datasetData
    
    return data
  }
  
  public func build() {
    guard complete == false else {
      print("CIFAR has already been loaded")
      return
    }
    
    let datasetData = get()
    data = datasetData
  }
  
  private func buildLabel(value: Int) -> [Float] {
    if !self.overrideLabel.isEmpty {
      return overrideLabel
    }
    
    guard value >= 0 else {
      return []
    }
    var labels = [Float].init(repeating: 0, count: 10)
    labels[value] = 1
    return labels
  }
  
  private func get() -> DatasetData {
    var data: [DatasetModel] = []
    
    for i in 1..<6 {
      let path = Bundle.module.path(forResource: "data_batch_\(i)", ofType: "bin")
      guard let path = path else {
        return ([],[])
      }
      
      let result = read(path: path, offset: Int(0), scaleBy: 1)
      
      let columns = 3072 + 1
      let rows = 1
      
      let shaped = result.reshape(columns: columns).batched(into: rows)
      
      shaped.forEach { imageRow in
        if let first = imageRow.first {
          var runningImage = first

          let label = Int(runningImage.removeFirst())
          
          let newRunningImage = runningImage.map { (Float($0) - 127.5) / 127.5 } // zero center
          
          if label == classType.rawValue {
            let reshapedRunningImage = newRunningImage.reshape(columns: 32).reshape(columns: 32)
            
            let image = Tensor(reshapedRunningImage)
            let labelArray = Tensor(buildLabel(value: label))
            
            let model = DatasetModel(data: image, label: labelArray)
            data.append(model)
          }
        }
      }
    }
  
    return (data, [])
  }

}
