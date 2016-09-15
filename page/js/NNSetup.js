function Net(size, weights, biases){
  this.sizes = size;
  this.weights = weights, this.biases = biases;
  //for (var i=0;i<weights.length; i++) this.weights.push(math.matrix(weights[i]));
  //for (var i=0;i<weights.length; i++) this.biases.push(math.matrix(biases[i]));
  this.feedforward = function(img) {
    var data = img;
    var a = data;
    for (var i=0;i<sizes.length-1;i++) {
      var z = [];
      for (var j=0; j<this.weights[i].length; j++) {
        var w_temp = math.transpose(math.matrix(this.weights[i][j]));
        pre_z = math.dot(w_temp, a);
        z.push(math.add(pre_z, this.biases[i][j]));
        //console.log(z);
      }
      a = [];
      for (var ind=0; ind<z.length; ind++) {
        a.push(1/(1+math.exp(-z[ind])));
      }
    }
    return a.indexOf(math.max(a));
  };
};

var sizes, weights, biases;
var json_req = $.getJSON( "https://raw.githubusercontent.com/AlexanderSwed/files/master/set.json", function(data) {
  sizes = data.sizes;
  weights = data.weights;
  biases = data.biases;
});
