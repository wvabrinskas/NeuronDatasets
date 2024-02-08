//
//  NSImage+Extensions.swift
//  GanTester (iOS)
//
//  Created by William Vabrinskas on 3/26/22.
//

import Neuron
#if os(macOS)
import Foundation
import Cocoa

public extension Float {
  var bytes: [UInt8] {
    withUnsafeBytes(of: self, Array.init)
  }
}

public extension NSImage {
  
  struct PixelData {
    var a: UInt8
    var r: UInt8
    var g: UInt8
    var b: UInt8
  }
  
  func asGrayScaleTensor(zeroCenter: Bool = false) -> Tensor {
    guard let pixelData = cgImage(forProposedRect: nil, context: nil, hints: nil)?.dataProvider?.data else { return Tensor() }

    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    var width: CGFloat = 0
    var height: CGFloat = 0
    
    representations.forEach { rep in
      width = max(CGFloat(rep.pixelsWide), width)
      height = max(CGFloat(rep.pixelsHigh), height)
    }
    
    var grayArray: [Float] = []

    for y in 0..<Int(height) {
      for x in 0..<Int(width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
        let r = Float(data[pixelInfo])
        let g = Float(data[pixelInfo + 1])
        let b = Float(data[pixelInfo + 2])
        
        var gray = (r + g + b) / 3

        if zeroCenter {
          gray = (gray - 127.5) / 127.5
        } else {
          gray = gray / 255.0
        }
        
        grayArray.append(gray)
      }
    }
    
    return Tensor([grayArray.reshape(columns: Int(width))])
  }
  
  func asRGBATensor(zeroCenter: Bool = false) -> Tensor {
    guard let pixelData = cgImage(forProposedRect: nil, context: nil, hints: nil)?.dataProvider?.data else { return Tensor() }
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    var rArray: [Float] = []
    var gArray: [Float] = []
    var bArray: [Float] = []
    var aArray: [Float] = []

    var width: CGFloat = 0
    var height: CGFloat = 0
    
    representations.forEach { rep in
      width = max(CGFloat(rep.pixelsWide), width)
      height = max(CGFloat(rep.pixelsHigh), height)
    }
    
    for y in 0..<Int(height) {
      for x in 0..<Int(width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
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
    
    let rawPixels = [rArray.reshape(columns: Int(width)),
                     gArray.reshape(columns: Int(width)),
                     bArray.reshape(columns: Int(width)),
                     aArray.reshape(columns: Int(width))]
    
    return Tensor(rawPixels)
  }
  
  
  func asRGBTensor(zeroCenter: Bool = false) -> Tensor {
    guard let pixelData = cgImage(forProposedRect: nil, context: nil, hints: nil)?.dataProvider?.data else {
      return Tensor()
    }
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    var rArray: [Float] = []
    var gArray: [Float] = []
    var bArray: [Float] = []
    
    var width: CGFloat = 0
    var height: CGFloat = 0
    
    representations.forEach { rep in
      width = max(CGFloat(rep.pixelsWide), width)
      height = max(CGFloat(rep.pixelsHigh), height)
    }
    
    for y in 0..<Int(height) {
      for x in 0..<Int(width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
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
    
    let rawPixels = [rArray.reshape(columns: Int(width)),
                     gArray.reshape(columns: Int(width)),
                     bArray.reshape(columns: Int(width))]
    
    return Tensor(rawPixels)
  }
  
  func asPixels() -> [PixelData] {
    var returnPixels = [PixelData]()
    
    guard let pixelData = self.cgImage(forProposedRect: nil, context: nil, hints: nil)?.dataProvider?.data else { return [] }
    
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
  
  static func from(_ pixels: [Float], size: (Int, Int)) -> NSImage? {
    let data: [UInt8] = pixels.map { UInt8(ceil(Double($0.isNaN ? 0 : $0) * 255)) }
    
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
    
    let image = context.makeImage().flatMap { NSImage(cgImage: $0, size: NSSize(width: CGFloat(width),
                                                                                height: CGFloat(height))) }
    
    return image
  }
  
  static func rawPixelsToPixelData(pixels: [UInt8], size: (width: Int, height: Int)) -> [PixelData] {
    guard pixels.isEmpty == false else { return [] }
    
    var pData: [PixelData] = []
    
    let stride = ((size.width * size.height) - 1)
    
    for i in 0..<Int(size.width * size.height) {
      let redP = pixels[i]
      let greenP = pixels[i + stride]
      let blueP = pixels[i + (2 * stride)]
      let pixelData = PixelData(a: 255, r: redP, g: greenP, b: blueP)
      pData.append(pixelData)
    }
    
    return pData
    
  }
  
  static func colorImage(_ pixels: [Float], size: (width: Int, height: Int), containsAlpha: Bool = false)-> NSImage? {
    let adjustedPixels: [UInt8] = pixels.map { UInt8(ceil(Double($0.isNaN ? 0 : $0) * 255)) }
    
    let pixelData = rawPixelsToPixelData(pixels: adjustedPixels, size: size)
    
    let width  = size.0
    let height = size.1
    
    guard width > 0 && height > 0 else { return nil }
    
    let pixelDataSize = MemoryLayout<PixelData>.size
    assert(pixelDataSize == 4)
    
    let data: Data = pixelData.withUnsafeBufferPointer {
      return Data(buffer: $0)
    }
    
    let cfdata = NSData(data: data) as CFData
    let provider: CGDataProvider! = CGDataProvider(data: cfdata)
    if provider == nil {
      print("CGDataProvider is not supposed to be nil")
      return nil
    }
    
    let cgimage: CGImage! = CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: width * pixelDataSize,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
      provider: provider,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent
    )
    if cgimage == nil {
      print("CGImage is not supposed to be nil")
      return nil
    }
    
    return NSImage(cgImage: cgimage, size: CGSize(width: width, height: height))
  }
}

#endif
