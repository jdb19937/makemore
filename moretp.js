function ispng(ar) {
  return (ar[0] == 0x89 && ar[1] == 0x50 && ar[2] == 0x4E && ar[3] == 0x47);
}

function moretpdecstr(str) {
  var enc = new TextEncoder('utf-8');
  return moretpdec(enc.encode(str))
}

function moretpencstr(words) {
  var dec = new TextDecoder('utf-8');
  return dec.decode(moretpenc(words))
}


function moretpenc(words) {
  var enc = new TextEncoder('utf-8');

  var binpart = new Array();
  var pwords = new Array();
  for (var i = 0; i < words.length; ++i) {
    var word = words[i];
    if (typeof(word) == 'string') {
      var morewords = word.split(/\s+/);
      for (var j = 0; j < morewords.length; ++j)
        pwords.push(morewords[j])
    } else {
      pwords.push("<" + word.length);
      for (var j = 0; j < word.length; ++j)
        binpart.push(word[j]);
    }
  }

  var line = enc.encode(pwords.join(' ') + "\n")

  var ret = new Uint8Array(line.length + binpart.length);
  for (var j = 0; j < line.length; ++j)
    ret[j] = line[j];
  for (var j = 0; j < binpart.length; ++j)
    ret[line.length + j] = binpart[j];

  return ret;
}


function moretpdec(inbuf) {
  var off;
  for (off = 0; off < inbuf.length; ++off) {
    if (inbuf[off] == 10) {
      break;
    }
  }
  if (off == inbuf.length)
    return null;

  var dec = new TextDecoder('utf-8');
  var cmdstr = dec.decode(inbuf.slice(0, off));
  ++off;

  var cmdwords = cmdstr.split(/\s+/);
  if (cmdwords.length > 0 && cmdwords[cmdwords.length - 1] == '') {
    cmdwords.pop();
  }

  var bytes = 0;
  for (var i = 0; i < cmdwords.length; ++i) {
    if (cmdwords[i][0] == '<') {
      bytes += parseInt(cmdwords[i].substr(1), 0);
    }
  }

  if (off + bytes > inbuf.length)
    return null;

  for (var i = 0; i < cmdwords.length; ++i) {
    if (cmdwords[i][0] == '<') {
      var b = parseInt(cmdwords[i].substr(1), 0);
      cmdwords[i] = inbuf.slice(off, off + b);

      if (ispng(cmdwords[i])) {
        var blob = new Blob([cmdwords[i]], {'type': 'image/png'});
        var url = URL.createObjectURL(blob); 
        cmdwords[i] = [url]
      } else {
        cmdwords[i] = dec.decode(cmdwords[i]);
      }

      off += b;
    }
  }

  return [cmdwords, off];
}

function concat(a, b) {
    var c = new (a.constructor)(a.length + b.length);
    c.set(a, 0);
    c.set(b, a.length);
    return c;
}

// https://stackoverflow.com/a/21394730
function sendMessage(sock, msg){
    // Wait until the state of the socket is not ready and send the message when it is...
    waitForSocketConnection(sock, function(){
        sock.send(msg);
    });
}

// Make the function wait until the connection is made...
function waitForSocketConnection(socket, callback){
    setTimeout(
        function () {
            if (socket.readyState === 1) {
                //window.console.log("connected")
                callback();
                return;

            } else {
                //window.console.log("connecting")
                waitForSocketConnection(socket, callback);
            }
        }, 200);
}


function moretpclient(onc) {
  this.inbuf = new Uint8Array(0)
  this.socket = new WebSocket('wss://peaple.io:3333/', ['binary']);
  this.socket.binaryType = 'arraybuffer';
  this.socket.inbuffer = new Uint8Array(0)

  var cli = this;
  this.socket.onmessage = function(m) {
    cli.moretpmessage(m)
  };

  if (!onc) { 
    this.socket.onopen = function() { alert('ready'); }
  } else {
    this.socket.onopen = onc;
  }

  this.markmap = { };
  this.mark = 0;
  this.asynccb = null;

  this.moretpreq = function(words, argmat, cb) {
    var markstr = "";
    markstr += this.mark;
    var nwords = [ ];

nwords.push("(");

    for (var a in argmat) {
      for (w in words) {
        nwords.push(words[w]);
      }
      var args = argmat[a]
      for (var b in args) {
        nwords.push(args[b]);
      }
      nwords.push(";");
    }
    for (w in words) {
      nwords.push(words[w]);
    }

nwords.push(")");
nwords.push("|");
nwords.push("pre");
nwords.push("out");
nwords.push(markstr);
nwords.push(";");

    nwords.push("echo");
    nwords.push("mark");
    nwords.push(markstr);

    var msg = moretpenc(nwords);
 
    this.markmap[markstr] = cb;
    this.mark++;

//    sendMessage(this.socket, msg)
this.socket.send(msg)
  }

  this.moretpline = function(line) {
    if (line[0] == "mark") {
      var mark = line[1];
      var cb = this.markmap[mark];
      cb.call(cb, null, mark);
      delete this.markmap[mark];
    } else if (line[0] == "out") {
      var mark = line[1];
      var cb = this.markmap[mark];
      var l2 = new Array(line.length - 1);
      for (var i = 2; i < line.length; ++i)
        l2[i - 2] = line[i];
      cb.call(cb, l2, mark)
    } else {
      if (this.asynccb) {
        this.asynccb.call(this.asynccb, line);
      }
    }
  }


  this.moretpmessage = function(m) {
    var x = new Uint8Array(m.data);

    this.inbuf = concat(this.inbuf, x);

    while (1) {
      var ret = moretpdec(this.inbuf);
      if (!ret) {
        break;
      }

      var msg = ret[0];

      this.moretpline(msg)

      var off = ret[1];
      this.inbuf = this.inbuf.slice(off, this.inbuf.length)
    }
  }
}

