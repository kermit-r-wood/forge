# Forge

[English](README_EN.md) | [中文](README.md)

## Introduction

Forge generates multi-color prints for FDM printers using color layering. It uses the natural translucency of filaments to create new colors by stacking standard ones. An algorithm calculates exactly how layers mix to match your target image with the filaments you have.

![Introduction](doc/intro_en.png)

## Features

*   **Runs Locally**: A standalone PySide app that runs on your machine.
*   **Zero Setup**: Download and run. No servers or complex installation.

## Download

Get the latest portable version from [Releases](../../releases).

## Usage

1.  **Import Image**: Drag your image into the app.
2.  **Configure Settings**: Set print size, layer height, and pick your filament colors.
3.  **Generate Model**: Click Generate. Forge calculates the layers and saves a 3MF model.
4.  **Slice and Print**: Open the 3MF in your slicer and print.

## Principle

Standard filaments like PLA and PETG are slightly translucent. When you stack different colors, they blend: a thinner red layer over white might look pink. Forge simulates this light transmission to calculate the filament stack needed to reproduce an image.
