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

public protocol TextSequenceDatasetSupporting: RawRepresentable<String>, CaseIterable {}

public typealias Header = Hashable & TextSequenceDatasetSupporting

/// Creates a dataset from a CSV file.
/// Set the K typealias to an `enum` that conforms to `Header`.
/// This typealias will be used to get the column of data you want from the CSV
public final class TextSequenceDataset<K: Header>: ProxyTokenizableDataset {
    
  public enum TextSequenceDatasetError: Error, LocalizedError {
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
    
  private let csvUrl: URL
  private let headerToFetch: K
  private let maxCount: Int
  private let cache: NSCache<NSString, NSArray> = .init()
  private var vectorizedAlready: [K: Bool] = [:]
  private let validationSplitPercentage: Tensor.Scalar
  private let filter: CharacterSet?
  private var maxLengthOfItem: Int = 0
  
  private enum CacheKey: NSString {
    case csv
    case header
  }
  
  public init(csvUrl: URL,
              headerToFetch: K,
              targetVocabSize: Int,
              maxCount: Int = 0, // 0 is all
              validationSplitPercentage: Tensor.Scalar = 0.2, // max is 0.9 and min is 0.1
              overrideLabel: [Tensor.Scalar] = [],
              filter: CharacterSet? = nil) {
    self.csvUrl = csvUrl
    self.headerToFetch = headerToFetch
    self.maxCount = maxCount
    self.validationSplitPercentage = max(min(0.9, validationSplitPercentage), 0.1)
    self.filter = filter
    
    super.init(tokenizer: .init(targetVocabSize: targetVocabSize),
               unitDataSize: .init(),
               overrideLabel: overrideLabel)
  }
  
  public required init(vectorizer: Vectorizer = .init(),
                       unitDataSize: Neuron.TensorSize,
                       overrideLabel: [Tensor.Scalar] = []) {
    fatalError("init(vectorizer:unitDataSize:overrideLabel:) has not been implemented")
  }
  
  public required init(tokenizer: Tokenizer,
                       unitDataSize: Neuron.TensorSize,
                       overrideLabel: [Tensor.Scalar] = []) {
    fatalError("init(tokenizer:unitDataSize:overrideLabel:) has not been implemented")
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
  
  // MARK: Private
  private func get() async throws {
    try fetchRawCSV()
    let csvData = try await getCSVData()
      
    let trainingSplit = Int(floor(Tensor.Scalar(csvData.count) * (1 - validationSplitPercentage)))
    let overrideLabelMap = overrideLabel.isEmpty ? nil : Tensor(overrideLabel.map { Tensor.Scalar($0) })
    
    func convertCSVData(_ d: (data: Tensor, label: Tensor)) -> DatasetModel {
      let input = d.data
      let label = d.label
                        
      return DatasetModel(data: input, label: overrideLabelMap ?? label)
    }
    
    let csvTrainingData = csvData[..<trainingSplit].map(convertCSVData(_:))
    let validationTrainingData = csvData[trainingSplit...].map(convertCSVData(_:))

    data = (csvTrainingData, validationTrainingData)
    complete = true
  }
  
  private func getCSVData() async throws -> [(data: Tensor, label: Tensor)] {
    var parsedCSV: [String]? = cache.object(forKey: CacheKey.csv.rawValue) as? [String]
    
    if parsedCSV == nil {
      parsedCSV = try fetchRawCSV()
    }
    
    guard var parsedCSV,
          let headers = parsedCSV[safe: 0]?.components(separatedBy: ",") else {
      throw TextSequenceDatasetError.headerMissing
    }
    
    let kHeaders = headers.map { K(rawValue: $0) }.compactMap { $0 }
    
    // drop headers
    let range: Range<Int> = maxCount <= 0 ? 0..<(parsedCSV.count - 1) : 0..<maxCount // - 1 because we removed the header
    parsedCSV = Array(Array(parsedCSV.dropFirst())[range]).filter({ $0.isEmpty == false })
    
    let inputStrings = parsedCSV.map { $0.components(separatedBy: ",") }
      .compactMap {
        $0[safe: kHeaders.firstIndex(of: self.headerToFetch) ?? 0]?.trimmingCharactersOptionally(in: self.filter)
      }
        
    // train the tokenizer on the input strings to build the tokens
    tokenizer.train(corpus: inputStrings)
    
    // One sequence length for the whole dataset: the RNN compiles a single `batchLength`.
    let length = sequenceLength(for: inputStrings, cappedAt: nil)
    
    // Labels are the inputs advanced by one token, so timestep i predicts token i + 1.
    let pairs = inputStrings.map { nextTokenPair(for: $0, sequenceLength: length) }

    return pairs
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


