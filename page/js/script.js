$(document).ready(function() {
  function imageToDataUri() {
    var img = document.getElementById('drawCanvas');
    var ctx = img.getContext('2d');
    var imageObj = document.getElementById('imageResized');
    imageObj.src = $('#drawCanvas').getCanvasImage('png');
    var img_r = document.getElementById('resizedCanvas');
    var ctx_r = img_r.getContext('2d');
    ctx_r.drawImage(imageObj, 0, 0, 28, 28);
    var imgData = ctx_r.getImageData(0, 0, 28, 28);
    var img_array = [];
    for (var i=0;i<imgData.data.length;i+=4)
    {
      if (imgData.data[i]<30 && imgData.data[i+1]<30 && imgData.data[i+2]<30 && imgData.data[i+3]>250) {
        img_array.push('1.0');
      }
      else img_array.push('0.0');
    }
    //
    //var blob = new Blob([img_array], {type: "application/json"});
    //var saveAs = window.saveAs;
    //saveAs(blob, "my_outfile.json");
    $('#drawCanvas').clearCanvas();
    $('#resizedCanvas').clearCanvas();
    return img_array;
}

  $('#saveBtn').on('click', function() {
    //this.href = $('#resizedCanvas').getCanvasImage('png');
    //this.download = 'image.png';
    //var photo = $('#drawCanvas').getCanvasImage().slice(22);
    img = imageToDataUri();
    var net = new Net(sizes, weights, biases);
    var predict = net.feedforward(img)
    $('#text').html(predict)
  });
});
