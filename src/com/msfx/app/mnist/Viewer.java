/*
 * Copyright (c) 2022 Miquel Sas.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.msfx.app.mnist;

import com.msfx.lib.fx.Alert;
import com.msfx.lib.fx.Buttons;
import com.msfx.lib.fx.Frame;
import com.msfx.lib.util.Files;
import com.msfx.lib.util.Numbers;
import com.msfx.lib.util.res.FileStringRes;
import com.msfx.lib.util.res.StringRes;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.Separator;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.stage.Stage;

import java.io.File;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

/**
 * A simple number image viewer.
 *
 * @author Miquel Sas
 */
public class Viewer extends Application {

	private class TaskDraw extends TimerTask {
		@Override
		public void run() { runTaskDraw(); }
	}

	public static void main(String[] args) { launch(args); }

	private Frame frame;
	private Canvas canvas;
	private Label label;
	private ProgressBar progressBar;

	private List<MNIST> images;
	private int imageIndex = 0;
	private final int pageSize = 100;

	private Timer timer;

	@Override
	public void start(Stage stage) throws Exception {

		/* Setup strings. */
		StringRes.setDefault(new FileStringRes("StringsLibrary.xml"));

		/* MNIST data type. */
		String res = queryDataType();
		if (res.equals("Cancel")) return;

		/* Image and label files. */
		String nameImages, nameLabels;
		if (res.equals("Train")) {
			nameImages = "train-images.idx3-ubyte.mnist_images";
			nameLabels = "train-labels.idx1-ubyte.mnist_labels";
		} else {
			nameImages = "t10k-images.idx3-ubyte.mnist_images";
			nameLabels = "t10k-labels.idx1-ubyte.mnist_labels";
		}
		File fileImages = Files.findFileWithinClassPathEntries(nameImages);
		File fileLabels = Files.findFileWithinClassPathEntries(nameLabels);
		if (fileLabels == null || fileImages == null) {
			return;
		}

		/* Read files. */
		images = MNIST.read(fileLabels, fileImages);

		/* Build frame. */

		frame = new Frame(stage);
		frame.getStage().setTitle("MNIST images viewer");
		double pad = frame.getPaneCombo().getPadding();

		canvas = new Canvas();
		Pane paneCanvas = new Pane(canvas);
		paneCanvas.setPrefWidth(15 * MNIST.COLS);
		paneCanvas.setPrefHeight(15 * MNIST.ROWS);
		canvas.widthProperty().bind(paneCanvas.widthProperty());
		canvas.heightProperty().bind(paneCanvas.heightProperty());
		canvas.widthProperty().addListener(l -> Platform.runLater(() -> drawImage()));
		canvas.heightProperty().addListener(l -> Platform.runLater(() -> drawImage()));

		label = new Label();
		label.setAlignment(Pos.CENTER);
		label.setMaxWidth(Double.MAX_VALUE);

		progressBar = new ProgressBar();
		progressBar.setMaxWidth(Double.MAX_VALUE);
		progressBar.setPrefHeight(10);
		progressBar.setProgress(0);

		frame.getPaneCombo().setTop(new Separator());
		frame.getPaneCombo().setCenter(paneCanvas);

		VBox vboxBottom = new VBox();
		vboxBottom.setPadding(new Insets(pad, pad, pad, pad));
		vboxBottom.getChildren().addAll(new Separator(), label, progressBar);
		frame.getPaneCombo().setBottom(vboxBottom);

		Button buttonRun = Buttons.button("run", "Start run images", false, false, false);
		Button buttonClose = Buttons.close(false, false, true);
		buttonRun.setOnKeyPressed(e -> onKeyPressed(e));
		buttonClose.setOnKeyPressed(e -> onKeyPressed(e));
		buttonClose.setOnAction(e -> {
			if (timer != null) {
				timer.cancel();
				timer = null;
			}
		});
		buttonRun.setOnAction(e -> runImages());

		frame.getPaneCombo().getButtonBar().getButtons().addAll(buttonRun, buttonClose);

		/* Key and mouse handlers. */
		frame.getScene().setOnKeyPressed(e -> onKeyPressed(e));
		frame.getScene().setOnScroll(e -> onScroll(e));

		frame.sizeAndCenter(0.4, 0.6);
		frame.show();
	}

	private void drawImage() {

		int imageIndex = this.imageIndex;
		MNIST image = images.get(imageIndex);
		double pad = frame.getPaneCombo().getPadding();

		double width = canvas.getWidth() - (2 * pad);
		double height = canvas.getHeight() - (2 * pad);

		double pixelWidth = width / MNIST.COLS;
		double pixelHeight = height / MNIST.ROWS;
		if (pixelWidth > pixelHeight) {
			pixelWidth = pixelHeight;
		}
		if (pixelHeight > pixelWidth) {
			pixelHeight = pixelWidth;
		}
		width = pixelWidth * MNIST.COLS;
		height = pixelHeight * MNIST.ROWS;

		double x0 = (canvas.getWidth() - width) / 2;
		double y0 = (canvas.getHeight() - height) / 2;

		int[][] bytes = image.getImage();

		GraphicsContext gc = canvas.getGraphicsContext2D();
		gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());

		// Save gc properties
		Paint saveFill = gc.getFill();
		double[] saveDashes = gc.getLineDashes();
		Paint saveStroke = gc.getStroke();
		gc.setStroke(Color.color(0.92, 0.92, 0.92));

		double x, y;

		y = y0;
		for (int r = 0; r < MNIST.ROWS; r++) {
			x = x0;
			for (int c = 0; c < MNIST.COLS; c++) {
				double b = ((double) bytes[r][c]) / 255d;
				Color color = new Color(b, b, b, 1.0);
				gc.setFill(color);
				gc.fillRect(x, y, pixelWidth, pixelHeight);
				x += pixelWidth;
			}
			y += pixelHeight;
		}

		x = x0;
		y = y0;
		for (int r = 0; r < MNIST.ROWS; r++) {
			gc.strokeLine(x, y, x + width, y);
			y += pixelHeight;
		}
		x = x0;
		y = y0;
		for (int c = 0; c < MNIST.COLS; c++) {
			gc.strokeLine(x, y, x, y + height);
			x += pixelWidth;
		}

		// Restore gc properties
		gc.setFill(saveFill);
		gc.setLineDashes(saveDashes);
		gc.setStroke(saveStroke);

		double index = imageIndex + 1;
		double size = images.size();
		double progress = index / size;
		progressBar.progressProperty().set(progress);

		StringBuilder text = new StringBuilder();
		text.append(image.getNumber());
		text.append(" (");
		text.append(imageIndex + 1);
		text.append("/");
		text.append(images.size());
		text.append(" - ");
		text.append(Numbers.getBigDecimal(progress * 100, 2).toPlainString());
		text.append("%)");
		label.textProperty().set(text.toString());
	}

	private void onKeyPressed(KeyEvent e) {
		if (timer != null) return;
		KeyCode keyCode = e.getCode();
		if (keyCode == KeyCode.DOWN) {
			if (imageIndex < images.size() - 1) {
				imageIndex++;
			}
		}
		if (keyCode == KeyCode.UP) {
			if (imageIndex > 0) {
				imageIndex--;
			}
		}
		if (keyCode == KeyCode.PAGE_UP) {
			if (imageIndex > 0) {
				imageIndex -= (imageIndex == 0 ? pageSize - 1 : pageSize);
				if (imageIndex < 0) {
					imageIndex = 0;
				}
			}
		}
		if (keyCode == KeyCode.PAGE_DOWN) {
			if (imageIndex < images.size() - 1) {
				imageIndex += (imageIndex == 0 ? pageSize - 1 : pageSize);
				if (imageIndex >= images.size()) {
					imageIndex = images.size() - 1;
				}
			}
		}
		if (keyCode == KeyCode.HOME) {
			imageIndex = 0;
		}
		if (keyCode == KeyCode.END) {
			imageIndex = images.size() - 1;
		}
		if (keyCode == KeyCode.ESCAPE) {
			System.exit(0);
		}
		Platform.runLater(() -> drawImage());
	}
	private void onScroll(ScrollEvent e) {
		if (timer != null) return;
		int scroll = (e.getTextDeltaY() < 0 ? 1 : -1);
		if (e.isControlDown()) {
			scroll *= (imageIndex == 0 ? pageSize - 1 : pageSize);
		}
		imageIndex += scroll;
		if (imageIndex < 0) {
			imageIndex = 0;
		}
		if (imageIndex >= images.size()) {
			imageIndex = images.size() - 1;
		}
		Platform.runLater(() -> drawImage());
	}

	private String queryDataType() {
		Alert alert = new Alert(0.4, 0.2);
		alert.setup(Alert.Type.PLAIN, Alert.Content.TEXT);
		alert.setTitle("MNIST data type");
		alert.addContentText("Please, indicate the type of MNIST data, train or test.");
		Button train = Buttons.button("train", "Train", true, false, true);
		Button test = Buttons.button("test", "Test", false, false, true);
		Button cancel = Buttons.button("cancel", "Cancel", false, true, true);
		alert.setButtons(train, test, cancel);

		Button result = alert.show();
		if (result == null) return "Cancel";
		if (result.isCancelButton()) return "Cancel";
		if (result.equals(train)) return "Train";
		return "Test";
	}

	private void runImages() {
		if (timer == null) {
			startRun();
		} else {
			stopRun();
		}
	}

	private void startRun() {
		Button buttonRun = (Button) frame.getScene().lookup("#run");
		buttonRun.setText("Stop run images");
		imageIndex = 0;
		timer = new Timer("RUN-IMAGES");
		timer.schedule(new TaskDraw(), 10, 1);
	}

	private void stopRun() {
		timer.cancel();
		timer = null;
		imageIndex = 0;
		Platform.runLater(() -> {
			Button buttonRun = (Button) frame.getScene().lookup("#run");
			buttonRun.setText("Start run images");
			drawImage();
		});
	}

	private void runTaskDraw() {
		Platform.runLater(() -> drawImage());
		imageIndex++;
		if (imageIndex >= images.size()) {
			stopRun();
		}
	}
}
