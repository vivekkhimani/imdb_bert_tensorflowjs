const express = require('express');
const request = require('request');
const bodyParser = require('body-parser');
const path = require('path');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const {loadGraphModel} = require('@tensorflow/tfjs-converter');
const sentence_tokenizer = require('sentence-tokenization');

const app = express();
const port = 3000;

app.use(express.static('public'));

var allowCrossDomain = function(req,res,next){
    res.header('Access-Control-Allow-Origin','*');
    res.header('Access-Control-Allow-Methods','GET,PUT,POST,DELETE');
    res.header('Access-Control-Allow-Headers','Content-Type');

    next();
}

app.use(allowCrossDomain);
app.use(bodyParser.urlencoded({
    extended:true
}
));

app.use(bodyParser.json());

//init tokenizer
const FullPairTokenizer = sentence_tokenizer.FullPairTokenizer;
const vocabFile = 'node_modules/sentence-tokenization/data/vocab.txt';

//tokenize string
function tokenizeInputs(htmlVal){
    const tokenizer = new FullPairTokenizer(vocabFile);
    const first = htmlVal;
    const second = '';
    const maxLen = 768;
    const tokens = tokenizer.tokenize(first,second,maxLen);
    tokens.pop();
    console.log(tokens);
    let segmentIds = tokenizer.convertTokensToSegments(tokens);
    let inputMask = tokenizer.convertTokensToMask(tokens);
    let inputIds = tokenizer.convertTokensToIds(tokens);
    [inputIds, segmentIds, inputMask] = tokenizer.padding(inputIds, segmentIds, inputMask, maxLen);
    return [inputIds, segmentIds, inputMask];
}

//make predictions
async function makePred(inputs){
    //load model
    const model = await tf.loadGraphModel("http://localhost:3000/model.json");
    console.log(model.executor._signature.inputs);

    /*
    //constant inputs
    const label_id = tf.tensor(0,[1],'int32');
    const filler = tf.scalar(0,'int32');

    //iterate through the obtained HTML values
    for (var i=0; i<original_length; i++){
        //var main_tokens = tokenizeInputs(str_array[i]);
        var main_tokens = tokenizeInputs('no_match no_match full-name tel email<input class="LoginInp" id="ContactPerson" name="ContactPerson" onkeyup="allowCharacter(this);" placeholder="Enter contact person name" type="text"/>');
        console.log(main_tokens);
        break;
        var inputI = main_tokens[0];
        var segmentI = main_tokens[1];
        var inputM = main_tokens[2];

        var inputMask = tf.tensor(inputM, [1,256], 'int32');
        var inputIds = tf.tensor(inputI, [1,256], 'int32');
        var segmentIds = tf.tensor(segmentI, [1,256], 'int32');

        var inputDict = {"label_ids:0" : label_id, "segment_ids:0" : segmentIds, "input_mask:0" : inputMask, "input_ids:0" : inputIds, "module/cls/predictions/output_bias:0" : filler, "module/cls/predictions/transform/dense/kernel:0" : filler, "module/cls/predictions/transform/LayerNorm/beta:0" : filler, "module/cls/predictions/transform/dense/bias:0" : filler, "module/cls/predictions/transform/LayerNorm/gamma:0": filler};

        var out = model.execute(inputDict,["loss/LogSoftmax:0"]);
        var out_index = tf.argMax(out,1).dataSync();
        var out_confidence = tf.max(out).dataSync();
        console.log("\n");
        console.log("Input:",str_array[i]);
        console.log("Predicted:",encoding.encodings[out_index[0]]);
        console.log("Log Softmax:",out_confidence[0].toString());

        //update previous labels
        previous_labels = updatePreviousLabels(previous_labels,encoding.encodings[out_index[0]]);
        str_arr[i+1] = appendPreviousLabels(previous_labels,str_arr[i+1]);
    	*/
    }
const inputs = "here";

makePred(inputs);
app.listen(port, () => console.log('Listening on port '+port));