//
//  QuickDrawDataset.swift
//  GanTester
//
//  Created by William Vabrinskas on 3/21/22.
//

import Foundation
import NumSwift
import Combine
import Logger
import Neuron

@available(macOS 12.0, *)
@available(iOS 15.0, *)
public class QuickDrawDataset: BaseDataset {
  
  private let trainingCount: Int
  private let validationCount: Int
  private var objectsToGet: [QuickDrawObject]
  private var zeroCentered: Bool
  private var useAllClasses: Bool
  
  
  /// Default initializer
  /// - Parameters:
  ///   - objectsToGet: The objects to get from the QuickDrawObject
  ///   - overrideLabel: Label to apply to every object
  ///   - useAllClasses: When set to `true` this will give each object a label with a one hot vector in its position in `QuickDrawObject`. When `false` the label will be based on the number of input objects to get. Default: `true`
  ///   - trainingCount: Total number of objects per `objectsToGet` to add to the training set
  ///   - validationCount: Total number of objects per `objectsToGet` to add to the validation set
  ///   - zeroCentered: Whether or not to zero center the dataset values. 0...1 or -1...1
  ///   - logLevel: The serverity level of logging.
  public init(objectsToGet: QuickDrawObject...,
              overrideLabel: [Tensor.Scalar] = [],
              useAllClasses: Bool = true,
              trainingCount: Int = 1000,
              validationCount: Int = 1000,
              zeroCentered: Bool = false,
              logLevel: LogLevel = .none) {
    self.trainingCount = trainingCount
    self.validationCount = validationCount
    self.objectsToGet = objectsToGet
    self.zeroCentered = zeroCentered
    self.useAllClasses = useAllClasses
    
    super.init(unitDataSize: .init(rows: 28, columns: 28, depth: 1),
               overrideLabel: overrideLabel)
    
    self.logLevel = logLevel
  }

  public override func build() async -> DatasetData {
    
    for i in 0..<objectsToGet.count {
      let objectToGet = objectsToGet[i]
      
      guard let path = objectToGet.url(), let url = URL(string: path) else {
        return data
      }
      
      let labelToUse: [Tensor.Scalar]
      
      if useAllClasses == false {
        var totalLabels: [Tensor.Scalar] = [Tensor.Scalar](repeating: 0, count: objectsToGet.count)
        totalLabels[i] = 1.0
        labelToUse = totalLabels
      } else {
        labelToUse = objectToGet.label()
      }
      
      let label = overrideLabel.isEmpty ? labelToUse : overrideLabel
      
      do {
        let urlRequest = URLRequest(url: url)
        self.log(type: .message, priority: .low, message: "Downloading dataset for \(objectToGet.rawValue)")
                 
        let download = try await URLSession.shared.data(for: urlRequest)
        let data = download.0
        
        let scale: Tensor.Scalar = zeroCentered ? 1 : 255
        
        var result: [Tensor.Scalar] = read(data: data, offset: 0x001a, scaleBy: scale)
        
        if zeroCentered {
          result = result.map { ($0 - 127.5) / 127.5 }
        }
        
        let shaped = result.reshape(columns: unitDataSize.columns).batched(into: unitDataSize.rows)
        
        self.log(type: .success, priority: .low, message: "Successfully donwloaded dataset - \(shaped.count) samples")

        let training = Array(shaped[0..<trainingCount]).map { DatasetModel(data: Tensor($0),
                                                                           label: Tensor(label)) }
        
        let validation = Array(shaped[trainingCount..<trainingCount + validationCount]).map { DatasetModel(data: Tensor($0),
                                                                                                           label: Tensor(label)) }
        
        self.data.training.append(contentsOf: training)
        self.data.val.append(contentsOf: validation)
        
      } catch {
        self.log(type: .error, priority: .alwaysShow, message: "Error getting dataset: \(error.localizedDescription)")
        print(error.localizedDescription)
      }
    }
        
    return await super.build()
  }
  
  public override func build() {
    Task {
      await build()
    }
  }

}
