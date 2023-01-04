//
//  UIImage+Extensions.swift
//  GanTester
//
//  Created by William Vabrinskas on 3/22/22.
//

import Neuron
#if os(iOS)
import Foundation
import UIKit
import NumSwift

public extension Float {
  var bytes: [UInt8] {
    withUnsafeBytes(of: self, Array.init)
  }
}


public extension UIImage {
  
  struct PixelData {
    var a: UInt8
    var r: UInt8
    var g: UInt8
    var b: UInt8
  }
  
  func asSquareRGBATensor(zeroCenter: Bool = false) -> Tensor {
    guard let pixelData = self.cgImage?.dataProvider?.data else { return Tensor() }
    
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    var rArray: [Float] = []
    var gArray: [Float] = []
    var bArray: [Float] = []
    var aArray: [Float] = []

    for y in 0..<Int(self.size.height) {
      for x in 0..<Int(self.size.width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
        var r = Float(data[pixelInfo])
        var g = Float(data[pixelInfo + 1])
        var b = Float(data[pixelInfo + 2])
        var a = Float(data[pixelInfo + 3])

        if zeroCenter {
          r = (r - 127.5) / 127.5
          g = (g - 127.5) / 127.5
          b = (b - 127.5) / 127.5
          a = (a - 127.5) / 127.5
        } else {
          r = r / 255.0
          g = g / 255.0
          b = b / 255.0
          a = a / 255.0
        }
        
        rArray.append(r)
        gArray.append(g)
        bArray.append(b)
        aArray.append(a)
      }
    }
    
    return Tensor([rArray.reshape(columns: Int(self.size.width)),
                   gArray.reshape(columns: Int(self.size.width)),
                   bArray.reshape(columns: Int(self.size.width)),
                   aArray.reshape(columns: Int(self.size.width))])
  }
  
  func asSquareRGBTensor(zeroCenter: Bool = false) -> Tensor {
    guard let pixelData = self.cgImage?.dataProvider?.data else { return Tensor() }
    
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    var rArray: [Float] = []
    var gArray: [Float] = []
    var bArray: [Float] = []
    
    for y in 0..<Int(self.size.height) {
      for x in 0..<Int(self.size.width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
        var r = Float(data[pixelInfo])
        var g = Float(data[pixelInfo + 1])
        var b = Float(data[pixelInfo + 2])
        
        if zeroCenter {
          r = (r - 127.5) / 127.5
          g = (g - 127.5) / 127.5
          b = (b - 127.5) / 127.5
        } else {
          r = r / 255.0
          g = g / 255.0
          b = b / 255.0
        }
        
        rArray.append(r)
        gArray.append(g)
        bArray.append(b)
      }
    }
    
    return Tensor([rArray.reshape(columns: Int(self.size.width)),
                   gArray.reshape(columns: Int(self.size.width)),
                   bArray.reshape(columns: Int(self.size.width))])
  }
  
  func asPixels() -> [PixelData] {
    var returnPixels = [PixelData]()
    
    guard let pixelData = self.cgImage?.dataProvider?.data else { return [] }
    
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    for y in 0..<Int(self.size.height) {
      for x in 0..<Int(self.size.width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
        let r = data[pixelInfo]
        let g = data[pixelInfo + 1]
        let b = data[pixelInfo + 2]
        let a = data[pixelInfo + 3]
        returnPixels.append(PixelData(a: a, r: r, g: g, b: b))
      }
    }
    return returnPixels
  }
  
  static func from(_ pixels: [Float], size: (Int, Int)) -> UIImage? {
    let data: [UInt8] = pixels.map { UInt8(ceil(Double($0) * 255)) }
    
    guard data.count >= 8 else {
      print("data too small")
      return nil
    }
    
    let width  = size.0
    let height = size.1
    
    let colorSpace = CGColorSpaceCreateDeviceGray()
    
    guard data.count >= width * height,
          let context = CGContext(data: nil,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: width,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.alphaOnly.rawValue),
          let buffer = context.data?.bindMemory(to: UInt8.self, capacity: width * height)
    else {
      return nil
    }
    
    for index in 0 ..< width * height {
      buffer[index] = data[index]
    }
    
    let image = context.makeImage().flatMap { UIImage(cgImage: $0) }
    
    return image
  }
  
}
#endif
