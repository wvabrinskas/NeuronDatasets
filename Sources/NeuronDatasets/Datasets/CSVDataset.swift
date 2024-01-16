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

public enum CSVType {
  case character, // where each line is a single word
       sentence // where each line is a full sentence
}

public protocol CSVSupporting: RawRepresentable<String>, CaseIterable {
  var type: CSVType { get }
  func order() -> [Self]
  func maxLengthOfItem() -> Int
}

public typealias Header = Hashable & CSVSupporting


/// Creates a dataset from a CSV file.
/// Set the K typealias to an `enum` that conforms to `Header`.
/// This typealias will be used to get the column of data you want from the CSV
public final class CSVDataset<K: Header>: BaseDataset, Logger, RNNSupportedDataset {

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
  
  private let csvUrl: URL
  private let parameters: Parameters
  private let vectorizer = Vectorizer<String>()
  private let headerToFetch: K
  private let maxCount: Int
  private let cache: NSCache<NSString, NSArray> = .init()
  private var vectorizedAlready: [K: Bool] = [:]
  private let validationSplitPercentage: Float
  private let labelOffset: Int
  private let filter: CharacterSet?
  
  private enum CacheKey: NSString {
    case csv
    case header
  }
  
  /// Default initializer
  /// - Parameters:
  ///   - csvUrl: the url of the CSV file
  ///   - headerToFetch: the K: Header value you want to fetch
  ///   - labelOffset: The amount to shift the label left by. eg. "mary." -> "ary..". Default: `1`. Useful in LSTM / RNN to predict next in sequence
  ///   - maxCount: the max number of objects you want. 0 = unlimited
  ///   - validationSplitPercentage: The validation split percentage to generate. min: 0.1, max: 0.9
  ///   - overrideLabel: the label to apply to each object. Otherwise the label will be set to the data. eg. `data: [0,1,0], label: [0,1,0]`
  ///   - parameters: The configuration parameters
  public init(csvUrl: URL,
              headerToFetch: K,
              maxCount: Int = 0, // 0 is all
              labelOffset: Int = 1,
              validationSplitPercentage: Float, // max is 0.9 and min is 0.1
              overrideLabel: [Float]? = nil,
              parameters: Parameters = .init(),
              filter: CharacterSet? = nil) {
    self.csvUrl = csvUrl
    self.parameters = parameters
    self.headerToFetch = headerToFetch
    self.maxCount = maxCount
    self.validationSplitPercentage = max(min(0.9, validationSplitPercentage), 0.1)
    self.labelOffset = labelOffset
    self.filter = filter
    
    super.init(unitDataSize: .init(), overrideLabel: overrideLabel ?? [])
  }
  
  public override func build() async -> DatasetData {
    do {
      try await get()
    } catch {
      print(error.localizedDescription)
    }
    return data
  }
  
  public override func build() {
    Task {
      do {
        try await get()
      } catch {
        print(error.localizedDescription)
      }
    }
  }
  
  public func getWord(for data: Tensor) -> [String] {
    if parameters.oneHot == false {
      let intArray = data.value.map { $0.map { $0.map { Int($0) }}}
      if let int = intArray[safe: 0]?[safe: 0] {
        return vectorizer.unvectorize(int)
      }
      
      return [""]
    } else {
      return vectorizer.unvectorizeOneHot(data)
    }
  }
  
  public func oneHot(_ items: [String]) -> Tensor {
    vectorizer.oneHot(items)
  }
  
  
  // MARK: Private
  private func get() async throws {
    try fetchRawCSV()
    let csvData = try await getCSVData()
    
    let trainingSplit = Int(floor(Float(csvData.count) * (1 - validationSplitPercentage)))
    let overrideLabelMap = overrideLabel.isEmpty ? nil : Tensor(overrideLabel.map { Tensor.Scalar($0) })
    
    let csvTrainingData = Array(csvData[..<trainingSplit]).map { d in
      let data = d
      var label = d
      
      let labelRaw = Array(label.value[labelOffset...])
      
      label = Tensor(labelRaw)
      
      if labelRaw.count < headerToFetch.maxLengthOfItem() {
        let delimiter = vectorizer.oneHot(["."])
        label = label.concat(delimiter, axis: 2)
      }
      
      return DatasetModel(data: data, label: overrideLabelMap ?? label)
    }
    
    let validationTrainingData = Array(csvData[trainingSplit...]).map { d in
      let data = d
      var label = d
      
      let labelRaw = Array(label.value[labelOffset...])
      
      label = Tensor(labelRaw)
      
      if labelRaw.count < headerToFetch.maxLengthOfItem() {
        let delimiter = vectorizer.oneHot(["."])
        label = label.concat(delimiter, axis: 2)
      }
      
      return DatasetModel(data: data, label: overrideLabelMap ?? label)
    }
    
    self.data = (csvTrainingData, validationTrainingData)
    complete = true
  }
  
  private func getCSVData() async throws -> [Tensor] {
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
        parsedCSV = Array(Array(parsedCSV.dropFirst())[range]).filter({ $0.isEmpty == false })
        
        let parsedByHeader = parsedCSV.map { $0.components(separatedBy: ",") }
          .compactMap {
            $0[safe: kHeaders.firstIndex(of: self.headerToFetch) ?? 0]?.trimmingCharactersOptionally(in: self.filter)
          }
        
//        if headerToFetch.type == .sentence {
//          let sentenceMap = parsedByHeader.compactMap { $0.split(separator: " ").map(String.init) }
//          let strings = sentenceMap.flatMap { $0 }
//          parsedByHeader = strings
//        }
        
        var result: [Tensor] = []
        
        parsedByHeader.forEach { string in
          let vector: [Tensor.Scalar]
          let stringToVectorize: [String]
          
          if headerToFetch.type == .character {
            stringToVectorize = string.fill(with: ".", max: headerToFetch.maxLengthOfItem()).characters
            vector = vectorizer.vectorize(stringToVectorize).map { $0.asTensorScalar }
          } else {
            var wordsInSentence = string.split(separator: " ").map(String.init)
            for i in 0..<headerToFetch.maxLengthOfItem() {
              if i >= wordsInSentence.count {
                wordsInSentence.append(".")
              } else if i % 2 != 0 {
                wordsInSentence.insert(" ", at: i)
              }
            }
            stringToVectorize = wordsInSentence
            vector = vectorizer.vectorize(wordsInSentence).map { $0.asTensorScalar }
          }
          
          if parameters.oneHot == false {
            result.append(Tensor(vector.map { [$0] }))
          } else {
            let oneHot = vectorizer.oneHot(stringToVectorize)
            result.append(oneHot)
          }
        }
                
        
        continuation.resume(returning: result)
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

fileprivate extension String {
  func trimmingCharactersOptionally(in filter: CharacterSet?) -> String {
    guard let filter else { return self }
    
    return trimmingCharacters(in: filter)
  }
}


