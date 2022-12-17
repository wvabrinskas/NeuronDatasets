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
public class QuickDrawDataset: Dataset, Logger {
  public var logLevel: LogLevel

  public var data: DatasetData = ([], []) {
    didSet {
      dataPassthroughSubject.send(data)
    }
  }
  
  public var overrideLabel: [Float]
  public let unitDataSize: TensorSize = .init(rows: 28, columns: 28, depth: 1)
  public var complete: Bool = false
  public var dataPassthroughSubject = PassthroughSubject<DatasetData, Never>()
  
  private let trainingCount: Int
  private let validationCount: Int
  private var objectToGet: QuickDrawObject
  private var zeroCentered: Bool
  
  public init(objectToGet: QuickDrawObject,
              overrideLabel: [Float] = [],
              trainingCount: Int = 1000,
              validationCount: Int = 1000,
              zeroCentered: Bool = false,
              logLevel: LogLevel = .none) {
    self.trainingCount = trainingCount
    self.validationCount = validationCount
    self.objectToGet = objectToGet
    self.zeroCentered = zeroCentered
    self.logLevel = logLevel
    self.overrideLabel = overrideLabel
  }

  public func build() async -> DatasetData {
    guard let path = objectToGet.url(), let url = URL(string: path) else {
      return data
    }
    
    let label = overrideLabel.isEmpty ? objectToGet.label() : overrideLabel
    
    do {
      let urlRequest = URLRequest(url: url)
      self.log(type: .message, priority: .low, message: "Downloading dataset for \(objectToGet.rawValue)")
               
      let download = try await URLSession.shared.data(for: urlRequest)
      let data = download.0
      
      let scale: Float = zeroCentered ? 1 : 255
      
      var result: [Float] = read(data: data, offset: 0x001a, scaleBy: scale)
      
      if zeroCentered {
        result = result.map { ($0 - 127.5) / 127.5 }
      }
      
      let shaped = result.reshape(columns: unitDataSize.columns).batched(into: unitDataSize.rows)
      
      self.log(type: .success, priority: .low, message: "Successfully donwloaded dataset - \(shaped.count) samples")

      let training = Array(shaped[0..<trainingCount]).map { DatasetModel(data: Tensor($0),
                                                                         label: Tensor(label)) }
      
      let validation = Array(shaped[trainingCount..<trainingCount + validationCount]).map { DatasetModel(data: Tensor($0),
                                                                                                         label: Tensor(label)) }
      
      self.data = (training, validation)
      return self.data
      
    } catch {
      self.log(type: .error, priority: .alwaysShow, message: "Error getting dataset: \(error.localizedDescription)")
      print(error.localizedDescription)
    }

    return ([], [])
  }
  
  public func build() {
    Task {
      await build()
    }
  }

}
