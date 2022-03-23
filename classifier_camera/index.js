let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const IMAGE_SIZE = 224;

function addExample(imgData, classId) {
    // Create image tensor
    const image = tf.browser.fromPixels(imgData);
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.

    const activation = net.infer(image, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    image.dispose();
    console.log("Image added");
}

function uploadFile() {
    var totalfiles = document.getElementById('files').files.length;

    if (totalfiles > 0) {
        for (var index = 0; index < totalfiles; index++) {
            var file = document.getElementById('files').files[index];
            var classId = document.getElementById('class').value;
            let reader = new FileReader();
            reader.onload = e => {
                // Fill the image & call predict.
                let img = document.createElement('img');
                img.src = e.target.result;
                img.width = IMAGE_SIZE;
                img.height = IMAGE_SIZE;
                img.onload = () => addExample(img, classId);
            };

            // Read in the image file as a data URL.
            reader.readAsDataURL(file);
        }
        alert(totalfiles + " images added for class " + classId);
        document.getElementById('class').value = parseInt(classId) + 1;
    } else {
        alert("Please select a file");
    }
}

async function app() {
    console.log('Loading mobilenet..');
    const webcam = await tf.data.webcam(webcamElement);

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');
    console.log('Backend: ' + tf.getBackend());
    var frames = 20;
    var fps = 0;
    var inferenceTime = 0;

    while (true) {
        if (classifier.getNumClasses() > 0) {
            const img = await webcam.capture();

            var startTime = performance.now();

            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(img, 'conv_preds');

            //console.log(activation);

            // Get the most likely class and confidence from the classifier module.
            const result = await classifier.predictClass(activation);

            var endTime = performance.now();

            if (frames == 20) {
                inferenceTime = endTime - startTime;
                frames = 0;
                fps = 1000 / inferenceTime;
            }

            frames++;
            var probability = Math.round(result.confidences[result.label] * 100) / 100;

            document.getElementById('console').innerText = `
                prediction: ${result.label}\n
                probability: ${probability.toFixed(2)}\n
                FPS: ${fps.toFixed(2)}
                time: ${inferenceTime.toFixed(2)}ms
            `;

            // Dispose the tensor to release the memory.
            img.dispose();
        }

        await tf.nextFrame();
    }
}

app();

