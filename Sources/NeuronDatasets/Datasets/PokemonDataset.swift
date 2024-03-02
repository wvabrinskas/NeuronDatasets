//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/2/24.
//

import Foundation
import NumSwift
import Combine
import Logger
import Neuron

@available(macOS 12.0, *)
@available(iOS 15.0, *)
public class PokemonDataset: BaseDataset, Logger {
  public var logLevel: LogLevel
  
  private let trainingCount: Int
  private let validationCount: Int
  private var zeroCentered: Bool
  
  private enum Link: String {
    case images = "https://www.dropbox.com/scl/fi/lqqdm77lovs9j8tjgxsr3/pokemon_images.npy?rlkey=nt6e3ii4pmc1bjv7rw4gbgcyg&dl=1"
    case labels = "https://www.dropbox.com/scl/fi/8ykgep7jmh2i43g22kyw4/pokemon_labels.npy?rlkey=d8qzhs6ycuxdh4iwrr7ktm20g&dl=1"
  }
  
  private let downloadUrlString = "https://www.dropbox.com/scl/fi/uadq8vx80ll3l97i7c82j/pokemon_dataset.npz?rlkey=qwslpw4fhjgyhfkld3rvh3a0e&dl=0"
  
  public init(overrideLabel: [Float] = [],
              trainingCount: Int = 1000,
              validationCount: Int = 1000,
              zeroCentered: Bool = false,
              logLevel: LogLevel = .none) {
    self.trainingCount = trainingCount
    self.validationCount = validationCount
    self.zeroCentered = zeroCentered
    self.logLevel = logLevel
    
    super.init(unitDataSize: .init(rows: 64, columns: 64, depth: 3),
               overrideLabel: overrideLabel)
  }

  public override func build() async -> DatasetData {
    guard let url = URL(string: Link.images.rawValue) else {
      return data
    }
    
    //let label = overrideLabel.isEmpty ? objectToGet.label() : overrideLabel
    let label = overrideLabel

    do {
      let urlRequest = URLRequest(url: url)
      self.log(type: .message, priority: .low, message: "Downloading dataset for Pokemon")
               
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
      return await super.build()
      
    } catch {
      self.log(type: .error, priority: .alwaysShow, message: "Error getting dataset: \(error.localizedDescription)")
      print(error.localizedDescription)
    }

    return ([], [])
  }
  
  public override func build() {
    Task {
      await build()
    }
  }

}
