/*
 * Copyright (c) 2022 Miquel Sas.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.msfx.app.mnist;

import com.msfx.lib.ml.data.Pattern;
import com.msfx.lib.ml.function.Normalizer;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * MNIST sample image data: a 28*28 byte image matrix and its number (0 to 9).
 *
 * @author Miquel Sas
 */
public class MNIST extends Pattern {

	public static List<MNIST> read(File fileLabels, File fileImages) throws IOException {
		List<MNIST> images = new ArrayList<>();

		var fisLbl = new FileInputStream(fileLabels);
		var bisLbl = new BufferedInputStream(fisLbl);
		var fisImg = new FileInputStream(fileImages);
		var bisImg = new BufferedInputStream(fisImg);

		/* Process image file up to the first data byte... */

		/* Skip magic number. */
		bisImg.skip(4);

		/* Read number of images. */
		var bytesNumImg = new byte[4];
		bisImg.read(bytesNumImg, 0, 4);
		var byteBufferNumImg = ByteBuffer.wrap(bytesNumImg);
		var numImages = byteBufferNumImg.getInt();

		/* Read number of rows. */
		var bytesNumRows = new byte[4];
		bisImg.read(bytesNumRows, 0, 4);
		var byteBufferNumRows = ByteBuffer.wrap(bytesNumRows);
		var numRows = byteBufferNumRows.getInt();

		/* Read number of columns. */
		var bytesNumColumns = new byte[4];
		bisImg.read(bytesNumColumns, 0, 4);
		var byteBufferNumColumns = ByteBuffer.wrap(bytesNumColumns);
		var numColumns = byteBufferNumColumns.getInt();

		/* Process label file up to the first data byte... */

		/* Skip magic number. */
		bisLbl.skip(4);

		/* Read number of labels. */
		var bytesNumLbl = new byte[4];
		bisLbl.read(bytesNumLbl, 0, 4);
		var byteBufferNumLbl = ByteBuffer.wrap(bytesNumLbl);
		var numLabels = byteBufferNumLbl.getInt();

		/* Check number of images versus labels. */
		if (numImages != numLabels) {
			bisImg.close();
			bisLbl.close();
			String err = "The number of images " + numImages;
			err += " has to be equals to the number of labels " + numLabels;
			throw new IOException(err);
		}

		/* Read data and create number images. */
		var imageSize = numRows * numColumns;
		for (int i = 0; i < numImages; i++) {
			var bytesRaw = new byte[imageSize];
			bisImg.read(bytesRaw, 0, imageSize);
			var number = bisLbl.read();
			var bytesInt = new int[imageSize];
			for (int j = 0; j < imageSize; j++) {
				bytesInt[j] = 255 - Byte.toUnsignedInt(bytesRaw[j]);
			}
			var image = new MNIST(number, bytesInt);
			images.add(image);
		}

		bisImg.close();
		bisLbl.close();

		return images;
	}

	/**
	 * Image rows.
	 */
	public static final int ROWS = 28;
	/**
	 * Image columns.
	 */
	public static final int COLS = 28;
	/**
	 * Number of input values of the pattern.
	 */
	public static final int IN = ROWS * COLS;
	/**
	 * Number of output values of the pattern.
	 */
	public static final int OUT = 10;

	/**
	 * The byte array of 28*28 = 784 elements.
	 */
	private final int[][] image;
	/**
	 * The number.
	 */
	private final int number;

	/**
	 * Constructor assigning the number and the bytes.
	 *
	 * @param number The represented number
	 * @param bytes  The raw bytes list
	 */
	public MNIST(int number, int[] bytes) {
		if (number < 0 || number > 9) {
			throw new IllegalArgumentException("Invalid number " + number);
		}
		if (bytes.length != ROWS * COLS) {
			throw new IllegalArgumentException("Invalid number of bytes per image " + bytes.length);
		}
		this.number = number;

		{
			image = new int[ROWS][COLS];
			int row = 0;
			int column = 0;
			for (int i = 0; i < bytes.length; i++) {
				image[row][column] = bytes[i];
				column++;
				if (column == COLS) {
					column = 0;
					row++;
				}
			}
		}

		{
			Normalizer normalizer = new Normalizer(255, 0, 1, 0);
			double[] inValues = new double[IN];
			int index = 0;
			for (int row = 0; row < ROWS; row++) {
				for (int col = 0; col < COLS; col++) {
					double imageByte = image[row][col];
					inValues[index++] = normalizer.normalize(imageByte);
				}
			}
			inputValues = new ArrayList<>();
			inputValues.add(inValues);
		}

		{
			double[] values = new double[OUT];
			int index = 0;
			for (int i = 0; i < number; i++) {
				values[index++] = 0.0;
			}
			values[index++] = 1.0;
			for (int i = number + 1; i < 10; i++) {
				values[index++] = 0.0;
			}
			outputValues = new ArrayList<>();
			outputValues.add(values);
		}
	}

	/**
	 * Returns the number.
	 *
	 * @return The number
	 */
	public int getNumber() { return number; }
	/**
	 * Returns the image as a two dimension byte array.
	 *
	 * @return The image
	 */
	public int[][] getImage() { return image; }

	/**
	 * Input values.
	 */
	private final List<double[]> inputValues;
	/**
	 * Output values.
	 */
	private final List<double[]> outputValues;

	/**
	 * Return the pattern input values.
	 *
	 * @return The pattern input values.
	 */
	@Override
	public List<double[]> getInputValues() { return inputValues; }
	/**
	 * Return the optional pattern output values.
	 *
	 * @return The pattern output values.
	 */
	@Override
	public List<double[]> getOutputValues() { return outputValues; }

	/**
	 * Returns the string representation.
	 */
	@Override
	public String toString() { return Integer.toString(getNumber()); }
}
