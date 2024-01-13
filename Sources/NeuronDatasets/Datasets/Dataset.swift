//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/10/22.
//

import Combine
import Foundation
import Neuron

public typealias DatasetData = (training: [DatasetModel], val: [DatasetModel])

/// The protocol that defines how to build a Neuron compatible dataset for training.
/// Example Datasets are:
/// ```
/// MNIST(only num: Int? = nil,
///       label: [Float] = [],
///       zeroCentered: Bool = false)
/// ```
/// ```
/// QuickDrawDataset(objectToGet: QuickDrawObject,
///                    label: [Float],
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
  var overrideLabel: [Float] { get set}

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

open class BaseDataset: Dataset {
  public var unitDataSize: Neuron.TensorSize = .init()
  public var data: DatasetData = ([], []) {
    didSet {
      dataPassthroughSubject.send(data)
    }
  }

  public var complete: Bool = false
  public var overrideLabel: [Float]
  public var dataPublisher:  AnyPublisher<DatasetData, Never> {
    dataPassthroughSubject.eraseToAnyPublisher()
  }
  
  private var dataPassthroughSubject: PassthroughSubject<DatasetData, Never> = .init()
  
  public init(unitDataSize: Neuron.TensorSize, 
              overrideLabel: [Float] = []) {
    self.unitDataSize = unitDataSize
    self.overrideLabel = overrideLabel
  }

  public func build() async -> DatasetData {
    // override
    ([],[])
  }
  
  public func build() {
    // override
  }
  
  public func trim(to: Int) {
   
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
}
