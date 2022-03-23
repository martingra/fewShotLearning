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

async function classifyImg(imgData) {
    // Create image tensor
    const image = tf.browser.fromPixels(imgData);
    var startTime = performance.now();

    // Get the activation from mobilenet from the image.
    const activation = net.infer(image, 'conv_preds');
    const result = await classifier.predictClass(activation);

    var endTime = performance.now();
    var inferenceTime = endTime - startTime;

    document.getElementById('console').innerText = `
        prediction: ${result.label}\n
        probability: ${Math.round(result.confidences[result.label] * 100) / 100}\n
        time: ${inferenceTime.toFixed(2)}ms
    `;

    // Dispose the tensor to release the memory.
    image.dispose();
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

function testImg() {
    if (classifier.getNumClasses() > 0) {

        var imgComp = document.getElementById("testImg");
        var file = document.getElementById("testImgFile").files[0];

        let reader = new FileReader();
        reader.onload = e => {
            // Fill the image & call predict.
            imgComp.src = e.target.result;
            imgComp.width = IMAGE_SIZE;
            imgComp.height = IMAGE_SIZE;
            imgComp.onload = () => classifyImg(imgComp);
        };

        reader.readAsDataURL(file);
    }
}

async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');
    console.log('Backend: ' + tf.getBackend());
}

app();

