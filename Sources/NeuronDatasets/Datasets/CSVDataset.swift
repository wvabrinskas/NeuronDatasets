//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/27/23.
//
import Combine
import Neuron
import Logger
import Foundation

public protocol CSVSupporting: RawRepresentable<String>, CaseIterable {
  func order() -> [Self]
  func maxLengthOfItem() -> Int
}

public typealias Header = Hashable & CSVSupporting

public final class CSVDataset<K: Header>: Dataset, Logger {
  public enum CSVDatasetError: Error, LocalizedError {
    case headerMissing
    case headerMappingError
    case couldNotMap

    public var errorDescription: String? {
      switch self {
      case .headerMissing:
        return NSLocalizedString("Could not find the header for the given csv", comment: "")
      case .headerMappingError:
        return NSLocalizedString("Could not map found headers to the given headers enum.", comment: "")
      case .couldNotMap:
        return NSLocalizedString("Could not map data to expected value", comment: "")
      }
    }
  }
  
  public struct Parameters {
    let oneHot: Bool
    
    public init(oneHot: Bool = true) {
      self.oneHot = oneHot
    }
  }
  
  public var logLevel: LogLevel = .high
  
  public var unitDataSize: TensorSize = .init()
  public var data: DatasetData = ([], []) {
    didSet {
      dataPassthroughSubject.send(data)
    }
  }
  public var complete: Bool = false
  public var dataPassthroughSubject: PassthroughSubject<DatasetData, Never> = .init()
  public var overrideLabel: [Float]
  
  private let csvUrl: URL
  private let parameters: Parameters
  private let vectorizer = Vectorizer<String>()
  private let headerToFetch: K
  private let maxCount: Int
  private let cache: NSCache<NSString, NSArray> = .init()
  private var vectorizedAlready: [K: Bool] = [:]
  private let validationSplitPercentage: Float
  
  private enum CacheKey: NSString {
    case csv
    case header
  }
  
  public init(csvUrl: URL,
              headerToFetch: K,
              maxCount: Int = 0, // 0 is all
              validationSplitPercentage: Float, // max is 0.9 and min is 0.1
              overrideLabel: [Float]? = nil,
              parameters: Parameters = .init()) {
    self.csvUrl = csvUrl
    self.overrideLabel = overrideLabel ?? []
    self.parameters = parameters
    self.headerToFetch = headerToFetch
    self.maxCount = maxCount
    self.validationSplitPercentage = max(min(0.9, validationSplitPercentage), 0.1)
  }
  
  public func build() async -> DatasetData {
    get()
    return data
  }
  
  public func build() {
    get()
  }
  
  // MARK: Private
  private func get() {
    do {
      try fetchRawCSV()
      let csvData = try getCSVData()
      
      let trainingSplit = Int(floor(Float(csvData.count) * (1 - validationSplitPercentage)))
      let overrideLabelMap = overrideLabel.isEmpty ? nil : overrideLabel.map { Tensor.Scalar($0) }
      
      let csvTrainingData = Array(csvData[..<trainingSplit]).map { DatasetModel(data: Tensor($0),
                                                                                label: Tensor($0)) }
      
      let validationTrainingData = Array(csvData[trainingSplit...]).map {DatasetModel(data: Tensor($0),
                                                                                      label: Tensor($0))}
      
      self.data = (csvTrainingData, validationTrainingData)
      complete = true
      
    } catch {
      print(error.localizedDescription)
    }
  }
  
  private func getCSVData() throws -> [[[Tensor.Scalar]]] {
    var parsedCSV: [String]? = cache.object(forKey: CacheKey.csv.rawValue) as? [String]
    
    if parsedCSV == nil {
      parsedCSV = try fetchRawCSV()
    }
    
    guard var parsedCSV,
          let headers = parsedCSV[safe: 0]?.components(separatedBy: ",") else {
      throw CSVDatasetError.headerMissing
    }
    
    let kHeaders = headers.map { K(rawValue: $0) }.compactMap { $0 }
    
    // drop headers
    let range: Range<Int> = maxCount <= 0 ? 0..<(parsedCSV.count - 1) : 0..<maxCount // - 1 because we removed the header
    parsedCSV = Array(Array(parsedCSV.dropFirst())[range])
    
    let parsedByHeader = parsedCSV.map { $0.components(separatedBy: ",") }
                                  .map { $0[kHeaders.firstIndex(of: headerToFetch) ?? 0] }
    

    let vectorized = parsedByHeader.map { vectorizer.vectorize($0.fill(with: ".",
                                                                       max: headerToFetch.maxLengthOfItem()).characters).map { $0.asTensorScalar } }
    
    if parameters.oneHot == false {
      return vectorized.map { [$0] }
    } else {
      let oneHotted = parsedByHeader.map { vectorizer.oneHot($0.fill(with: ".",
                                                                     max: headerToFetch.maxLengthOfItem()).characters)}
      
      return oneHotted
    }
  }
  
  @discardableResult
  private func fetchRawCSV() throws -> [String] {
    if let cached = cache.object(forKey: CacheKey.csv.rawValue) as? [String] {
      return cached
    }
    
    let content = try String(contentsOfFile: csvUrl.absoluteString)
    let parsedCSV = content.components(separatedBy: "\n")
    
    cache.setObject(NSArray(array: parsedCSV), forKey: CacheKey.csv.rawValue)
    
    return parsedCSV
  }

}
