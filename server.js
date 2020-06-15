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