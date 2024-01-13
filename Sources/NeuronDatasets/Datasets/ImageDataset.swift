//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/2/23.
//

import Combine
import Foundation
import Neuron
import Logger
#if os(macOS)
import Cocoa
#endif
#if os(iOS)
import UIKit
#endif

/// Creates an RGB dataset from a directory of images. Alpha is removed.
public class ImageDataset: BaseDataset, Logger {
  public enum ImageDatasetError: Error, LocalizedError {
    case imageDepthError
    
    public var errorDescription: String? {
      switch self {
      case .imageDepthError:
        return NSLocalizedString("Expected image depth should be equal to the image size depth", comment: "")
      }
    }
  }
  
  public enum ImageDepth: CaseIterable {
    case rgb, rgba, grayScale
    
    var expectedDepth: Int {
      switch self {
      case .rgb:
        return 3
      case .rgba:
        return 4
      case .grayScale:
        return 1
      }
    }
  }
  
  public var logLevel: LogLevel = .low
  
  private let imagesDirectory: String
  private let zeroCentered: Bool
  private let maxCount: Int
  private let validationSplitPercent: Float
  private let imageDepth: ImageDepth
  
  /// Initializes an RGB ImageDataset. This call throws an error if the
  /// - Parameters:
  ///   - imagesDirectory: The directory of the images to load. All images should be the same size.
  ///   - imageSize: The expected size of the images
  ///   - label: The label to apply to every image.
  ///   - imageDepth: ImageDepth that describes the expected depth of the images.
  ///   - maxCount: Max count to add to the dataset. Could be useful to save memory. Setting it to 0 will use the whole dataset.
  ///   - validationSplitPercent: Number between 0 and 1. The lower the number the more likely it is the image will be added to the training dataset otherwise it'll be added to the validation dataset.
  ///   - zeroCentered: Format image RGB values between -1 and 1. Otherwise it'll be normalized to between 0 and 1.
  public init(imagesDirectory: URL,
              imageSize: CGSize,
              label: [Float],
              imageDepth: ImageDepth,
              maxCount: Int = 0,
              validationSplitPercent: Float = 0,
              zeroCentered: Bool = false) {

    self.imagesDirectory = imagesDirectory.path
    self.zeroCentered = zeroCentered
    self.maxCount = maxCount
    self.validationSplitPercent = validationSplitPercent
    self.imageDepth = imageDepth
    
    super.init(unitDataSize: TensorSize(rows: Int(imageSize.width),
                                        columns: Int(imageSize.height),
                                        depth: imageDepth.expectedDepth),
               overrideLabel: label)
  }
  
  public override func build() async -> DatasetData {
    guard complete == false else {
      print("ImageDataset has already been loaded")
      return self.data
    }
    
    readDirectory()
    return self.data
  }
  
  public override func build() {
    readDirectory()
  }
  
  private func getImageTensor(for url: String) -> Tensor {
    guard let rawUrl = URL(string: url) else { return Tensor() }
    
    #if os(macOS)
    if let image = NSImage(contentsOf: rawUrl) {
      switch imageDepth {
      case .rgb:
        return image.asRGBTensor(zeroCenter: zeroCentered)
      case .rgba:
        return image.asRGBATensor(zeroCenter: zeroCentered)
      case .grayScale:
        return image.asGrayScaleTensor(zeroCenter: zeroCentered)
      }
    }
    #elseif os(iOS)
    if let image = UIImage(contentsOfFile: url) {
      switch imageDepth {
      case .rgb:
        return image.asRGBTensor(zeroCenter: zeroCentered)
      case .rgba:
        return image.asRGBATensor(zeroCenter: zeroCentered)
      case .grayScale:
        return image.asGrayScaleTensor(zeroCenter: zeroCentered)
      }
    }
    #endif
    
    return Tensor()
  }
 
  private func readDirectory() {
    guard complete == false else {
      print("ImageDataset has already been loaded")
      return
    }
    
    do {
      let contents = try FileManager.default.contentsOfDirectory(atPath: imagesDirectory)
      
      var training: [DatasetModel] = []
      var validation: [DatasetModel] = []
      
      let maximum = maxCount == 0 ? contents.count : maxCount
      
      for index in 0..<maximum {
        let imageUrl = contents[index]
        let path = "file://" + imagesDirectory.appending("/\(imageUrl)")
        let imageData = getImageTensor(for: path)
        let label = Tensor(overrideLabel)
        if Float.random(in: 0...1) >= validationSplitPercent {
          training.append(DatasetModel(data: imageData, label: label))
        } else {
          validation.append(DatasetModel(data: imageData, label: label))
        }
      }
      
      self.data = (training, validation)
      complete = true
    } catch {
      print(error.localizedDescription)
    }
  }
}
