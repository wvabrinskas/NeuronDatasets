//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/10/22.
//

import Combine
import Foundation
import Neuron
import Logger

@propertyWrapper
public struct Percentage<Value: FloatingPoint> {
  private var value: Value
  
  public init(wrappedValue: Value) {
    self.value = wrappedValue
  }
  
  public var wrappedValue: Value {
    get {
      return max(min(value, 1), 0)
    }
    set {
      value = max(min(newValue, 1), 0)
    }
  }
}


public typealias DatasetData = (training: [DatasetModel], val: [DatasetModel])

/// The protocol that defines how to build a Neuron compatible dataset for training.
/// Example Datasets are:
/// ```
/// MNIST(only num: Int? = nil,
///       label: [Tensor.Scalar] = [],
///       zeroCentered: Bool = false)
/// ```
/// ```
/// QuickDrawDataset(objectToGet: QuickDrawObject,
///                    label: [Tensor.Scalar],
///                    trainingCount: Int = 1000,
///                    validationCount: Int = 1000,
///                    zeroCentered: Bool = false,
///                    logLevel: LogLevel = .none)
/// ```
public protocol Dataset: AnyObject {
  var unitDataSize: TensorSize { get }
  /// The resulting dataset
  var data: DatasetData { get }
  /// Indicator that the dataset has loaded
  var complete: Bool { get }
  /// Combine publisher for the dataset
  var dataPublisher: AnyPublisher<DatasetData, Never> { get }
  /// Override the label that the dataset provides
  var overrideLabel: [Tensor.Scalar] { get set}

  /// Read the dataset file from a path
  /// - Parameters:
  ///   - path: Path to the dataset
  ///   - offset: Address offset in bytes to start reading the data from. Many datasets put stuff like size in the first few bytes of the binary dataset file.
  ///   - scaleBy: Value to divide each datapoint by.
  /// - Returns: The dataset as an array of FloatingPoint values
  func read<T: FloatingPoint>(path: String, offset: Int, scaleBy: T) -> [T]
  /// Read the dataset file from a path
  /// - Parameters:
  ///   - data: Data object of the dataset
  ///   - offset: Address offset in bytes to start reading the data from. Many datasets put stuff like size in the first few bytes of the binary dataset file.
  ///   - scaleBy: Value to divide each datapoint by.
  /// - Returns: The dataset as an array of FloatingPoint values
  func read<T: FloatingPoint>(data: Data, offset: Int, scaleBy: T) -> [T]

  /// Returns a an array of UInt8 values to project a bitmap
  /// - Parameters:
  ///   - path: Path to the dataset
  ///   - offset: Address offset in bytes to start reading the data from. Many datasets put stuff like size in the first few bytes of the binary dataset file.
  /// - Returns: Array of UInt8 values to project as a bitmap
  func bitmap(path: String, offset: Int) -> [UInt8]
  /// Returns a an array of UInt8 values to project a bitmap
  /// - Parameters:
  ///   - data: Data object of the dataset
  ///   - offset: Address offset in bytes to start reading the data from. Many datasets put stuff like size in the first few bytes of the binary dataset file.
  /// - Returns: Array of UInt8 values to project as a bitmap
  func format(data: Data, offset: Int) -> [UInt8]

  /// Async/Await compatable build operation that will read and return the dataset.
  /// - Returns: The `DatasetData` object
  func build() async -> DatasetData

  /// Builds the dataset and will publish it to the Combine publisher and downstream subscribers.
  func build()
  
  /// Trims the dataset to specified count
  func trim(to: Int)
}

public protocol DatasetMergable: Dataset {
  func merge(with dataset: Self)
}

open class BaseDataset: Dataset, Logger {
  public var logLevel: LogLevel = .low
  
  public var unitDataSize: Neuron.TensorSize = .init()
  public var data: DatasetData = ([], []) {
    didSet {
      dataPassthroughSubject.send(data)
    }
  }

  public var complete: Bool = false
  public var overrideLabel: [Tensor.Scalar]
  public var dataPublisher:  AnyPublisher<DatasetData, Never> {
    dataPassthroughSubject.eraseToAnyPublisher()
  }
  
  private var dataPassthroughSubject: PassthroughSubject<DatasetData, Never> = .init()
  private var trimTo: Int?
  
  public init(unitDataSize: Neuron.TensorSize, 
              overrideLabel: [Tensor.Scalar] = []) {
    self.unitDataSize = unitDataSize
    self.overrideLabel = overrideLabel
  }

  public func build() async -> DatasetData {
    // override
    trim()
    randomize()
    return data
  }
  
  public func build() {
    // override
    trim()
    randomize()
  }
  
  /// Trims the dataset to set amount
  public func trim(to: Int) {
    guard to > 0 else { return }
    trimTo = to
  }

  public func format(data: Data, offset: Int) -> [UInt8] {
    let array = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [UInt8] in
      return Array<UInt8>(pointer[offset..<pointer.count])
    }

    return array
  }

  public func bitmap(path: String, offset: Int) -> [UInt8] {
    let url = URL(fileURLWithPath: path)

    do {
      let data = try Data(contentsOf: url)
      return format(data: data, offset: offset)
    } catch {
      print(error.localizedDescription)
      return []
    }
  }

  public func read<T: FloatingPoint>(data: Data, offset: Int, scaleBy: T) -> [T] {
    let bitmap = format(data: data, offset: offset)
    let result = bitmap.map { T($0) / scaleBy }
    return result
  }

  public func read<T: FloatingPoint>(path: String, offset: Int, scaleBy: T) -> [T] {
    let bitmap = bitmap(path: path, offset: offset)
    let result = bitmap.map { T($0) / scaleBy }
    return result
  }
  
  func merge(with dataset: BaseDataset) {
    guard dataset.unitDataSize == unitDataSize else { return }
    
    data.training.append(contentsOf: dataset.data.training)
    data.val.append(contentsOf: dataset.data.val)

    randomize()
  }
  
  func randomize() {
    data.training = data.training.shuffled()
    data.val = data.val.shuffled()
  }
  
  private func trim() {
    guard let trimTo else { return }
    
    var training = data.training
    var validation = data.val
    
    if training.count > trimTo {
      training = Array(training[0..<trimTo])
    }
    
    if validation.count > trimTo {
      validation = Array(validation[0..<trimTo])
    }
                
    data = (Array(training), Array(validation))
  }
}
