function previewImage() {
    var oFReader = new FileReader();
    oFReader.readAsDataURL(document.getElementById("upload-file").files[0]);

    oFReader.onload = function (oFREvent) {
        document.getElementById("blurred-image").src = oFREvent.target.result;
    };
};

let model;

async function loadModel() {
    model = await tf.loadLayersModel('./web/model/model.json');
    console.log("Successfully loaded model.");
}

function runPrediction() {

}


loadModel();
