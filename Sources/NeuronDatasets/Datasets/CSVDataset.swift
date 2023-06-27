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


/// Creates a dataset from a CSV file.
/// Set the K typealias to an `enum` that conforms to `Header`.
/// This typealias will be used to get the column of data you want from the CSV
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
  
  /// Default initializer
  /// - Parameters:
  ///   - csvUrl: the url of the CSV file
  ///   - headerToFetch: the K: Header value you want to fetch
  ///   - maxCount: the max number of objects you want. 0 = unlimited
  ///   - validationSplitPercentage: The validation split percentage to generate. min: 0.1, max: 0.9
  ///   - overrideLabel: the label to apply to each object. Otherwise the label will be set to the data. eg. `data: [0,1,0], label: [0,1,0]`
  ///   - parameters: The configuration parameters
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
    do {
      try await get()
    } catch {
      print(error.localizedDescription)
    }
    return data
  }
  
  public func build() {
    Task {
      do {
        try await get()
      } catch {
        print(error.localizedDescription)
      }
    }
  }
  
  // MARK: Private
  private func get() async throws {
    try fetchRawCSV()
    let csvData = try await getCSVData()
    
    let trainingSplit = Int(floor(Float(csvData.count) * (1 - validationSplitPercentage)))
    let overrideLabelMap = overrideLabel.isEmpty ? nil : Tensor(overrideLabel.map { Tensor.Scalar($0) })
    
    let csvTrainingData = Array(csvData[..<trainingSplit]).map { DatasetModel(data: Tensor($0),
                                                                              label: overrideLabelMap ?? Tensor($0)) }
    
    let validationTrainingData = Array(csvData[trainingSplit...]).map {DatasetModel(data: Tensor($0),
                                                                                    label: overrideLabelMap ?? Tensor($0))}
    
    self.data = (csvTrainingData, validationTrainingData)
    complete = true
  }
  
  private func getCSVData() async throws -> [[[Tensor.Scalar]]] {
    return try await withCheckedThrowingContinuation { continuation in
      Task {
        var parsedCSV: [String]? = cache.object(forKey: CacheKey.csv.rawValue) as? [String]
        
        if parsedCSV == nil {
          do {
            parsedCSV = try fetchRawCSV()
          } catch {
            continuation.resume(throwing: error)
          }
        }
        
        guard var parsedCSV,
              let headers = parsedCSV[safe: 0]?.components(separatedBy: ",") else {
          continuation.resume(throwing: CSVDatasetError.headerMissing)
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
          continuation.resume(returning: vectorized.map { [$0] })
          
        } else {
          let oneHotted = parsedByHeader.map { vectorizer.oneHot($0.fill(with: ".",
                                                                         max: headerToFetch.maxLengthOfItem()).characters)}
          
          continuation.resume(returning: oneHotted)
        }
      }
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
