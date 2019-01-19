#!/usr/bin/perl

use FindBin qw($Bin);
use lib "$Bin";
use FunnyName qw(genname);

my $NONCE = int rand(1<<31);

my $README;
open(my $fp, "cat $Bin/README |fmt -100 |") or die "no README";
{
  undef local $/;
  $README = <$fp>;
}
close($fp);

my $NAME;
if (my $path = $ENV{PATH_INFO}) {
  $path =~ /^\/([a-z_][a-z0-9_]+$)/ or die;
  $NAME = $1;
  length($name) < 32 or die;
} else {
  $NAME = genname();
  print "Location: http://localhost/makemore/peaple.cgi/$NAME\r\n\r\n";
  exit 0;
}

print "Content-type: text/html; charset=utf-8\r\n";
print "\r\n";

my %sub = (
  'README'	=> $README,
  'NAME'	=> $NAME,
  'NONCE'	=> $NONCE
);

while (<DATA>) {
  s/\$([A-Z]+)/$sub{$1}/eg;
  print;
}

__DATA__
<html> <head> <title>makemore peaple v0.1</title>
<base href="..">

<style>
input[type="file"] {
    display: none;
}
.custom-file-upload {
    border: 1px solid #ccc;
    background-color: white;
    display: inline-block;
    padding: 1px 6px;
    text-align: center;
    cursor: pointer;
    font-size: small;
    width: 150px;
}
.noselect {
  -webkit-touch-callout: none; /* iOS Safari */
    -webkit-user-select: none; /* Safari */
     -khtml-user-select: none; /* Konqueror HTML */
       -moz-user-select: none; /* Firefox */
        -ms-user-select: none; /* Internet Explorer/Edge */
            user-select: none; /* Non-prefixed version, currently
                                  supported by Chrome and Opera */
}

body {
  font-family: courier;
}

.attrclass {
  padding-top: 1px;
  padding-left: 3px;
  font-family: courier;
  font-size: small
}


</style>
</head>

<!--
-->

<body bgcolor="darkgray" class="noselect">

<table width=1200 cellpadding=4 cellspacing=0 border=3 bordercolor=gray>
<tr>
<td align=left valign=center style='border: 0; font-size: xx-large'>
<b>$NAME</b>
</td>

<td align=right valign=top style='border: 0'><b><a href="https://github.com/jdb19937/makemore">makemore</a> peaple v0.1</b><br/>by Dan Brumleve</td>
</tr>
</table>

<table width=1200 cellpadding=4 cellspacing=0 border=0>
<tr>
<td width=512>
<img width=512 height=512 id="profile" src="profile.cgi/$NAME.jpg?nonce=$NONCE" style="image-rendering: pixelated">
</td>

<td style='background-color: lightgray; border: 0' valign="top">

<table width=900 height=320 style="border:0; overflow: hidden" cellspacing=0>
<tr><td height=160px></td></tr>
<tr><td colspan=8><hr/><font size=+2><b>frens of $NAME</b></font></td><tr>
<tr valign=top height=110>
<td id=fren0 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren1 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren2 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren3 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren4 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren5 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren6 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren7 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
</tr>
<tr valign=top height=110>
<td id=fren8 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren9 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren10 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren11 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren12 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren13 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren14 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
<td id=fren15 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: small"> </td>
</tr>

<tr>

  <td colspan=8>
    <hr/>
    <input type=text id=frenbuf name=frenbuf onClick="this.select();" onChange="frenbufhilite()" onKeyDown="frenbufhilite()" oninput="frenbufhilite()" onpaste="frenbufhilite()" value="" size=32 maxlength=32/>
    <input type=button onClick="document.getElementById('frenbuf').value = gennom(); frenbufhilite()" value="gen nom"/>
    <input type=button onClick="addfren()" value="add fren"/>
    <input type=button onClick="setparsontarget()" value="set target"/>
    <input type=button onClick="gotofile()" value="goto file"/>
    <input type=button onClick="makebread()" value="bread"/>
    <input type=button onClick="cloneto()" value="clone to"/>
    <input type=button onClick="document.getElementById('frenbuf').select(); document.execCommand('copy')" value="copy nom"/>

  </td>
</tr>
</table>


</td>
</tr></table>


<br/><br/>


<table height="5" width=1"><tr><td></td></tr></table>


<!--
<table width=1200 cellpadding=4 cellspacing=0 border=3 bordercolor=gray><tr>
<td align=left style='border: 0'><b>$NAME</b>'s maintenance panel</td>
</tr></table>
<table height="5" width=1"><tr><td></td></tr></table>
-->





<table width=1200 cellpadding=0 cellspacing=0><tr><td>

<div id="attrcon" style="border-width: 3px; border-color: gray; border-style: solid">

<table cellpadding=0 cellspacing=1><tr>

<tr>



<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr19' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> high_cheekbones  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr29' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> rosy_cheeks  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr25' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> oval_face  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr26' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> pale_skin  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr24' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> beard  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr0' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> stubble  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr30' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> sideburns  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr16' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> goatee  </td></tr></table>   </td>
</tr><tr>


<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr22' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> mustache  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr12' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> bushy_eyebrows  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr1' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> arched_eyebrows  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr23' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> narrow_eyes  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr28' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> balding  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr4' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> bald  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr32' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> straight_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr33' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> wavy_hair  </td></tr></table>   </td>
</tr><tr>

<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr5' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> bangs  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr8' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> black_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr9' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> blonde_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr11' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> brown_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr17' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> gray_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr13' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> chubby  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr14' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> double_chin  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr21' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> mouth_open  </td></tr></table>   </td>
</tr><tr>

<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr20' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> male  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr39' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> young  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr10' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> blurry  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr2' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> attractive  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr3' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> tired  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr6' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> big_lips  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr7' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> big_nose  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr27' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> pointy_nose  </td></tr></table>   </td>
</tr><tr>


<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr31' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> smiling  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr15' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> eyeglasses  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr18' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> makeup  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr34' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> earrings  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr35' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> hat  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr36' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> lipstick  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr37' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> necklace  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr38' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> necktie  </td></tr></table>   </td>
</tr><tr>

<tr><td colspan=8 border=1></td></tr>

<script>
function vis(id) { document.getElementById(id).style.visibility = 'visible' }
function unvis(id) { document.getElementById(id).style.visibility = 'hidden' }
</script>

<td colspan=2> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='tag0' style='border: 3px solid gray; background-color: black; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> <input id="tag0in" size="24" maxlength=24 value="" onFocus="vis('taghint')" onFocusout="unvis('taghint')"/>  </td></tr></table>   </td>

<td colspan=2> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='tag1' style='border: 3px solid gray; background-color: black; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> <input id="tag1in" size="24" maxlength=24 value="" onFocus="vis('taghint')" onFocusout="unvis('taghint')"/>  </td></tr></table>   </td>
<td id=taghint colspan=4 style='visibility: hidden'> [enter any two tags matching regex /^[a-z][a-z0-9_]*$/] </td>

</table>


</div>

</td>
<td valign=top>

<table cellpadding=0 cellspacing=0>
<tr>
  <td width=64 valign=top>
    <div id="attrstatic" style="border-width: 3px; border-color: #606010; border-style: solid; width: 64; height: 64">
      <img src="static.png" onMouseDown="
doubleclick('attrstatic', genrandomattrs, 'attrcon')
"/>
    </div>
  </td>
</tr><tr>
<!--
  <td width=64>
    <div id="attrperson" style="border-width: 3px; border-color: #606010; border-style: solid; width: 64; height: 64">
      <img src="person.png"  onMouseDown="doubleclick('attrperson', genpersonattrs, 'attrcon')  "/>
    </div>
  </td>
-->
</tr>
</table>

</td></tr>

</table>
<table height="5" width=1"><tr><td></td></tr></table>




<table cellpadding=0 cellspacing=1 width=1200 style='border-width: 3px; border-color: gray; border-style: solid'>
  <tr>
    <td width=64>
      <div id="colorshower" style="border-width: 3px; border-color: gray; border-style: solid; width: 64; height: 64; background-color: black" onClick="changetool('d')"> </div>
    </td>
    <td width=64>
      <div id="colorpicker" style="border-width: 3px; border-color: gray; border-style: solid; width: 64; height: 64;"> <img src="spectrum.png" onClick="changetool('p')"> </div>
    </td>
    <td width=64>
      <div id="blurtool" style="border-width: 3px; border-color: gray; border-style: solid; width: 64; height: 64;"> <img src="blur.png" onClick="changetool('b')"> </div>
    </td>
    <td width=64>
      <div id="sharptool" style="border-width: 3px; border-color: gray; border-style: solid; width: 64; height: 64;"> <img src="sharp.png" onClick="changetool('s')">
    </td>
    <td>
      <div id="fuzztool" style="border-width: 3px; border-color: gray; border-style: solid; width: 64; height: 64;"> <img src="fuzz.png" onClick="changetool('f')"> </div>
    </td>

  <td align=left>
    <input type=button onclick="requpdate()" value="request update"/>

    <input type=button id=ctrlockbutton onClick="lockallcontrols()" value="lock all controls">
    <input type=button id=tgtlockbutton onClick="lockalltargets()" value="lock all targets">
    <input type=button id=ctrlockbutton onClick="unlockallcontrols()" value="unlock all controls">
    <input type=button id=ctrlockbutton onClick="unlockalltargets()" value="unlock all targets">
  <input type="button" for="imageLoader" value="upload target image" onClick="getElementById('imageLoader').click()"/>
<input type="file" size="60" id="imageLoader" name="imageLoader" accept="image/png, image/jpeg"/>
<div style="display: none"><canvas id="imageCanvas" width=64 height=64></canvas></div>
  </td>


  <td></td>

  <td width=64>
    <div id="burnin" style="border-width: 3px; border-color: #606010; border-style: solid; width: 64; height: 64">
      <img src="burn.png" onMouseDown="doubleclick2('burnin', doburnin) "/>
    </div>
  </td>


  </tr>

</table>
<table height="5" width=1"><tr><td></td></tr></table>






<table width=1200 cellpadding=0 cellspacing=4>

  <tr>
  <th style='font-size: small'>generated from controls</th>
  <th style='font-size: small'>palette | generated from prev target</th>
  <th style='font-size: small'>target</th>
  <th style='font-size: small'>adjustment</th>
  <th style='font-size: small'>controls</th>
  <th style='font-size: small'>functions</th>
  <th></th>

  <tr>

 <td width=320 style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px"><canvas id="stage1gen" width=320 height=320></canvas></td>
 <td width=320 style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px"><canvas id="palette" onClick="clickpalette(event)" width=320 height=320></canvas></td>
 <td width=320 style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px"><canvas id="stage1" width=320 height=320></canvas></td>

 <td width=320 id="stage1adjborder" style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">
    <canvas id="stage1adj" width=320 height=320
onClick="toggletgtlock(1)"
    ></canvas>
  </td>

  <td width=64 valign=top>

  <table cellspacing=0 cellpadding=0>
  <tr>
  <td width=64 id="stage1conborder" style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">
    <canvas id="stage1con" width=64 height=320 onClick="togglectrlock(1)">
</canvas>
  </td></tr></table>




  </td>
  <td valign=top>
    <table cellspacing=4 cellpadding=0>
    <tr><td><input style="width: 100px" type="button" value="scramble" onClick="requpdatecon(1, 1)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="tone up" onClick="requpdatecon(1, 2)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="tone down" onClick="requpdatecon(1, 3)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="lock controls" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="unlock controls" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="lock target" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="unlock target" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="recombine" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="blend" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="burn in" onClick=""/></td></tr>
    </table>
  </td>
</tr><tr>





  <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage2gen" width=320 height=320></canvas></td>
  <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage2orig" width=320 height=320></canvas></td>
  <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage2" width=320 height=320></canvas></td>
 <td width=320 id="stage2adjborder" style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">
    <canvas id="stage2adj" width=320 height=320
onClick="toggletgtlock(2)"
    ></canvas>
  </td>


  <td valign=top>
  <table cellspacing=0 cellpadding=0><tr>
  <td width=64 id="stage2conborder" style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">
    <canvas id="stage2con" width=64 height=320 onClick="togglectrlock(2)">
</canvas>
  </td></tr></table>
  </td>

  <td valign=top>
    <table cellspacing=4 cellpadding=0>
    <tr><td><input style="width: 100px" type="button" value="scramble" onClick="requpdatecon(2, 1)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="tone up" onClick="requpdatecon(2, 2)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="tone down" onClick="requpdatecon(2, 3)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="lock controls" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="unlock controls" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="lock target" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="unlock target" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="recombine" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="blend" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="burn in" onClick=""/></td></tr>
    </table>
  </td>
</tr>

<!--
</table>
<br/>
<table>
-->

<tr>
 <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage3gen" width=320 height=320></canvas></td>
  <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage3orig" width=320 height=320></canvas></td>
  <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage3" width=320 height=320></canvas></td>
 <td width=320 id="stage3adjborder" style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">
    <canvas id="stage3adj" width=320 height=320
onClick="toggletgtlock(3)"
    ></canvas>
  </td>

<td valign=top>
  <table cellspacing=0 cellpadding=0><tr valign=top>
  <td width=64 id="stage3conborder" style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">
    <canvas id="stage3con" width=64 height=320 onClick="togglectrlock(3)">
</canvas>
  </td></tr></table>
</td>

  <td valign=top>
    <table cellspacing=4 cellpadding=0>
    <tr><td><input style="width: 100px" type="button" value="scramble" onClick="requpdatecon(3, 1)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="tone up" onClick="requpdatecon(3, 2)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="tone down" onClick="requpdatecon(3, 3)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="lock controls" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="unlock controls" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="lock target" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="unlock target" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="recombine" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="blend" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="burn in" onClick=""/></td></tr>
    </table>
  </td>





</tr>

<!--
</table>
<br/>
<table>
-->

<tr>

 <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage4gen" width=320 height=320></canvas></td>
  <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage4orig" width=320 height=320></canvas></td>
  <td style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px" width=320><canvas id="stage4" width=320 height=320></canvas></td>
 <td width=320 id="stage4adjborder" style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">
    <canvas id="stage4adj" width=320 height=320
onClick="toggletgtlock(4)"
    ></canvas>
  </td>
<td valign=top>
  <table cellspacing=0 cellpadding=0><tr valign=top>
  <td width=64 id="stage4conborder" style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">
    <canvas id="stage4con" width=64 height=320 onClick="togglectrlock(4)">
</canvas>
  </td></tr></table>
</td>
  <td valign=top>
    <table cellspacing=4 cellpadding=0>
    <tr><td><input style="width: 100px" type="button" value="scramble" onClick="requpdatecon(4, 1)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="tone up" onClick="requpdatecon(4, 2)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="tone down" onClick="requpdatecon(4, 3)"/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="lock controls" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="unlock controls" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="lock target" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="unlock target" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="recombine" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="blend" onClick=""/></td></tr>
    <tr><td><input style="width: 100px" type="button" value="burn in" onClick=""/></td></tr>
    </table>
  </td>
</tr>

</table>

<br/>



<br/>


<table bgcolor=white cellpadding=16><tr><td>
<h1>instructions for makemore peaple v0.1</h1>
<font size=+2>
<pre>
$README
</pre>
</font>
</td></tR></table>


</body>

 <script>

function gennom(gen) {  
  var pre1 = [
     'jay', 'crys', 'jun', 'holl', 'harv', 'thund', 'bowl', 'haz', 'zinc', 'surf',
     'dun', 'jol', 'mar', 'fren', 'frod', 'froom',
     'grun', 'herk', 'jark', 'kev', 'lem', 'mel', 'norm',
    'ned', 'rugg', 'frap', 'sorr', 'trop', 'vern', 'wum',
    'bob', 'frood'
  ];
  
  var msuf1 = ['io', 'ius', 'ader', 'on', 'son', 'y', 'athan', 'opher'];
  var fsuf1 = ['ia', 'ina', 'ona', 'adra', 'iona', 'i'];
  
  var pre2 = [
    'frep', 'mup', 'tran', 'string', 'synth',
    'dun', 'sami', 'gork', 'free', 'blar', 'gan', 'net',
    'fron', 'long', 'tarp', 'lor', 'norm', 'neur', 'blur',
    'nimble', 'luck', 'just', 'fair', 'pool', 'hoop', 'milk'
  ];
  
  var suf2 = [
    'bean', 'stone', 'barn', 'tree',
    'rock', 'screen', 'fake', 'null', 'grond', 'bun',
    'nam', 'plain', 'real', 'sticker', 'picker',
    'packer', 'macker', 'maker', 'bane', 'son',
    'wood', 'bull', 'ster',
    'friend', 'raver', 'rover', 'mark', 'hawk', 'razor', 'weed', 'walker',
    'burger'
  ]
    
  var pick = function(x) { return x[Math.floor(Math.random() * x.length)] }

  var p1 = pick(pre1)
  var s1
  if (gen == 'm') {
    s1 = pick(msuf1)
  } else if (gen == 'f') {
    s1 = pick(fsuf1)
  } else {
    s1 = pick(pick([msuf1, fsuf1]))
  }

  var p2 = pick(pre2)
  if (Math.random() < 0.05) {
    p2 = "mc" + p2
  }
  if (Math.random() < 0.03) {
    p2 = pick(["von_", "van_", "van_der_", "al", "o_"]) + p2
  }
  var s2 = pick(suf2)
    
  return p1 + s1 + "_" + p2 + s2
}


function frenbufhilite() {
  var nom = document.getElementById('frenbuf').value
  for (var i = 0; i < 15; ++i) {
    var frentd = document.getElementById('fren' + i)
    frentd.style.borderColor = (nom == window.frens[i] ? 'blue' : 'gray');
  }
}

function addfren() {
  var nom = document.getElementById('frenbuf').value
  if (nom == '') {
    nom = gennom()
  }

  window.mynewfren = nom

  requpdate()

  window.mynewfren = ''

  document.getElementById('frenbuf').value = ''
  frenbufhilite()
  return true
}

function cloneto() {
  var hyper = new Uint8Array(8);
  hyper[0] = 4

  var nom = document.getElementById('frenbuf').value
  if (nom == '') {
    return false
  }

  window.dirtyfren = nom
  window.mynewfren = nom
  document.getElementById('frenbuf').value = ''

  requpdate(hyper)

  window.mynewfren = ''
  frenbufhilite()
  return true
}

function makebread() {
  var hyper = new Uint8Array(8);
  hyper[0] = 5

  var nom = document.getElementById('frenbuf').value
  if (nom == '') {
    return false
  }

  window.dirtyfren = nom
  window.mynewfren = nom
  //document.getElementById('frenbuf').value = ''

  requpdate(hyper)

  window.mynewfren = ''
  //frenbufhilite()
  return true
}


// https://stackoverflow.com/a/10906961
function handleImage(e){
var canvas = document.getElementById('imageCanvas');
var ctx = canvas.getContext('2d');
    var reader = new FileReader();
    reader.onload = function(event){
        var img = new Image();
        img.onload = function(){
            ctx.drawImage(img,0,0,64,64);
            var rgbdata = ctx.getImageData(0, 0, 64, 64).data;
            applyloaded(rgbdata)
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(e.target.files[0]);     
}

function gotofile() {
  var frenbuf = document.getElementById("frenbuf");
  if (frenbuf.value == "") {
    return;
  }
  window.location = "peaple.cgi/" + frenbuf.value
}

function setparsontarget() {
  var canvas = document.getElementById('imageCanvas');
  var ctx = canvas.getContext('2d');
  var img = new Image();
  img.onload = function(){
      ctx.drawImage(img,0,0,64,64);
      var rgbdata = ctx.getImageData(0, 0, 64, 64).data;
      applyloaded(rgbdata)
  }
  var frenbuf = document.getElementById("frenbuf");
  if (frenbuf.value != "") {
    img.src = "profile.cgi/" + frenbuf.value + "?nonce=" + Math.floor(Math.random() * 1000000);
  }
}

function applyloaded(rgbdata) {
  var stage4 = document.getElementById('stage4');
  var adjctx = document.getElementById('stage4adj').getContext('2d')
  var scale = stage4.scale
  var oldlabdata = window.gendata;

  for (var j = 0, k = 0; j < 64*64*3; j += 3, k += 4) {
    var col = [rgbdata[k+0], rgbdata[k+1], rgbdata[k+2]]
    var newlab = rgbtolab(col)

    var oldlab = [oldlabdata[j+0], oldlabdata[j+1], oldlabdata[j+2]]

    var adjlabcol = [
      128 + (newlab[0] - oldlab[0]) / 2.0,
      128 + (newlab[1] - oldlab[1]) / 2.0,
      128 + (newlab[2] - oldlab[2]) / 2.0
    ];

    stage4.adjlab[j+0] = adjlabcol[0]
    stage4.adjlab[j+1] = adjlabcol[1]
    stage4.adjlab[j+2] = adjlabcol[2]

    var px = Math.floor((j % (64 * 3)) / 3)
    var py = Math.floor(j / (64 * 3));
    var adjcol = labtorgb(adjlabcol)
    adjctx.fillStyle = mkcol(adjcol)
    adjctx.fillRect(px * scale, py * scale, scale, scale);
  }

  var stage3 = document.getElementById('stage3');
  adjctx = document.getElementById('stage3adj').getContext('2d')
  scale = stage3.scale
  for (var j = 0; j < 32*32*3; j += 3) {
    var adjlabcol = [128,128,128]

    stage3.adjlab[j+0] = adjlabcol[0]
    stage3.adjlab[j+1] = adjlabcol[1]
    stage3.adjlab[j+2] = adjlabcol[2]

    var px = Math.floor((j % (32 * 3)) / 3)
    var py = Math.floor(j / (32 * 3));
    var adjcol = labtorgb(adjlabcol)
    adjctx.fillStyle = mkcol(adjcol)
    adjctx.fillRect(px * scale, py * scale, scale, scale);
  }

  var stage2 = document.getElementById('stage2');
  adjctx = document.getElementById('stage2adj').getContext('2d')
  scale = stage2.scale
  for (var j = 0; j < 16*16*3; j += 3) {
    var adjlabcol = [128,128,128]

    stage2.adjlab[j+0] = adjlabcol[0]
    stage2.adjlab[j+1] = adjlabcol[1]
    stage2.adjlab[j+2] = adjlabcol[2]

    var px = Math.floor((j % (16 * 3)) / 3)
    var py = Math.floor(j / (16 * 3));
    var adjcol = labtorgb(adjlabcol)
    adjctx.fillStyle = mkcol(adjcol)
    adjctx.fillRect(px * scale, py * scale, scale, scale);
  }

  var stage1 = document.getElementById('stage1');
  adjctx = document.getElementById('stage1adj').getContext('2d')
  scale = stage1.scale
  for (var j = 0; j < 8*8*3; j += 3) {
    var adjlabcol = [128,128,128]

    stage1.adjlab[j+0] = adjlabcol[0]
    stage1.adjlab[j+1] = adjlabcol[1]
    stage1.adjlab[j+2] = adjlabcol[2]

    var px = Math.floor((j % (8 * 3)) / 3)
    var py = Math.floor(j / (8 * 3));
    var adjcol = labtorgb(adjlabcol)
    adjctx.fillStyle = mkcol(adjcol)
    adjctx.fillRect(px * scale, py * scale, scale, scale);
  }

  window.tgtlock = 15

  requpdate()
}



function namebuf() {
  var myname = '$NAME'
  var buf = new Uint8Array(32)
  buf.fill(0)
  if (myname.length > 31) {
    return buf
  }
  for (var i = 0; i < myname.length; ++i) {
    buf[i] = myname.charCodeAt(i)
  }
  return buf
}

var n_stages = 4

// https://github.com/antimatter15/rgb-lab/blob/master/color.js

function labtorgb(lab){
  var L = lab[0]
  var A = lab[1]
  var B = lab[2]

  if (L < 0) { L = 0; } else if (L > 255) { L = 255; }
  if (A < 0) { A = 0; } else if (A > 255) { A = 255; }
  if (B < 0) { B = 0; } else if (B > 255) { B = 255; }

  L = L / 2.55
  A = A / 2.55
  B = B / 2.55

  A -= 50
  B -= 50

  var y = (L + 16.0) / 116.0;
  var x = A / 500.0 + y;
  var z = y - B / 200.0;
  var r, g, b;

  var x3 = x * x * x
  var y3 = y * y * y
  var z3 = z * z * z
  x = 0.95047 * ((x3 > 0.008856) ? x3 : (x - 16.0/116.0) / 7.787);
  y = 1.00000 * ((y3 > 0.008856) ? y3 : (y - 16.0/116.0) / 7.787);
  z = 1.08883 * ((z3 > 0.008856) ? z3 : (z - 16.0/116.0) / 7.787);

  r = x *  3.2406 + y * -1.5372 + z * -0.4986;
  g = x * -0.9689 + y *  1.8758 + z *  0.0415;
  b = x *  0.0557 + y * -0.2040 + z *  1.0570;


  r = (r > 0.0031308) ? (1.055 * Math.pow(r, 1.0/2.4) - 0.055) : 12.92 * r;
  g = (g > 0.0031308) ? (1.055 * Math.pow(g, 1.0/2.4) - 0.055) : 12.92 * g;
  b = (b > 0.0031308) ? (1.055 * Math.pow(b, 1.0/2.4) - 0.055) : 12.92 * b;


  return [Math.max(0, Math.min(1, r)) * 255.0, 
          Math.max(0, Math.min(1, g)) * 255.0, 
          Math.max(0, Math.min(1, b)) * 255.0]
}


function rgbtolab(rgb){
  var r = rgb[0] / 255,
      g = rgb[1] / 255,
      b = rgb[2] / 255,
      x, y, z;

  if (r < 0) { r = 0; } if (r > 1) { r = 1; }
  if (g < 0) { g = 0; } if (g > 1) { g = 1; }
  if (b < 0) { b = 0; } if (b > 1) { b = 1; }

  r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
  g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
  b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

  x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
  y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
  z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

  x = (x > 0.008856) ? Math.pow(x, 1/3) : (7.787 * x) + 16/116;
  y = (y > 0.008856) ? Math.pow(y, 1/3) : (7.787 * y) + 16/116;
  z = (z > 0.008856) ? Math.pow(z, 1/3) : (7.787 * z) + 16/116;

  var L = (116 * y) - 16
  var A = 500 * (x-y)
  var B = 200 * (y-z)

  A += 50
  B += 50

  return [Math.max(0, Math.min(100, L)) * 2.55, 
          Math.max(0, Math.min(100, A)) * 2.55, 
          Math.max(0, Math.min(100, B)) * 2.55]
}

function gencontrols(stage, nc, dev) {
  var canvas = document.getElementById(stage + "con")
  canvas.cbuf = new Float64Array(nc)
  for (var c = 0; c < nc; ++c) {
    var r = gaussianRand(function() { return Math.random() }) * dev
    var s = 256.0 / (1.0 + Math.exp(-r))
    if (s > 255) s = 255; if (s < 0) s = 0
    canvas.cbuf[c] = s
  }
  fillbw(canvas, 16)
  requpdate()
}

function tgtlockon(stage) {
  var adjcanvas = document.getElementById("stage" + stage + "adjborder")

  adjcanvas.style.borderColor = 'blue'
  window.tgtlock |= (1 << (stage - 1));
}


function toggletgtlock(stage) {
  var adjcanvas = document.getElementById("stage" + stage + "adjborder")

  if (window.tgtlock & (1 << (stage - 1))) {
    adjcanvas.style.borderColor = 'gray'
    window.tgtlock &= ~(1 << (stage - 1));

    if (!(window.ctrlock & (1 << (stage - 1)))) {
      window.ctrlock |= (1 << (stage - 1));
    }
  } else {
    adjcanvas.style.borderColor = 'blue'
    window.tgtlock |= (1 << (stage - 1));
  }

  requpdate();
}

function togglectrlock(stage) {
  var concanvas = document.getElementById("stage" + stage + "conborder")

  if (window.ctrlock & (1 << (stage - 1))) {
    concanvas.style.borderColor = 'gray'
    window.ctrlock &= ~(1 << (stage - 1));

    if (!(window.tgtlock & (1 << (stage -  1)))) {
      window.tgtlock |= (1 << (stage - 1));
    }
  } else {
    concanvas.style.borderColor = 'blue'
    window.ctrlock |= (1 << (stage - 1));
  }

  requpdate();
}

function lockallcontrols() {
  var changed = 0
  for (var stage = 1; stage <= 4; ++stage) {
    if (!(window.ctrlock & (1 << (stage - 1)))) {
      window.ctrlock |= (1 << (stage - 1));
      ++changed
    }
  }

  if (changed) {
    requpdate()
  }
}

function unlockallcontrols() {
  var changed = 0
  for (var stage = 1; stage <= 4; ++stage) {
    if (window.ctrlock & (1 << (stage - 1))) {
      ++changed
      window.ctrlock &= ~(1 << (stage - 1));

      if (!(window.tgtlock & (1 << (stage -  1)))) {
        window.tgtlock |= (1 << (stage - 1));
      }
    }
  }

  if (changed) {
    requpdate()
  }
}

function lockalltargets() {
  var changed = 0
  for (var stage = 1; stage <= 4; ++stage) {
    if (!(window.tgtlock & (1 << (stage - 1)))) {
      window.tgtlock |= (1 << (stage - 1));
      ++changed
    }
  }

  if (changed) {
    requpdate()
  }
}

function unlockalltargets() {
  var changed = 0
  for (var stage = 1; stage <= 4; ++stage) {
    if (window.tgtlock & (1 << (stage - 1))) {
      window.tgtlock &= ~(1 << (stage - 1));
      ++changed

      if (!(window.ctrlock & (1 << (stage - 1)))) {
        window.ctrlock |= (1 << (stage - 1));
      }
    }
  }

  if (changed) {
    requpdate()
  }
}
  

function clearadj(stage) {
  var canvas = document.getElementById("stage" + stage)
  canvas.adjlab.fill(128)
  var adjcanvas = document.getElementById("stage" + stage + "adj")
  var adjctx = adjcanvas.getContext('2d');
  adjctx.fillStyle = '#808080'
  adjctx.fillRect(0, 0, adjcanvas.width, adjcanvas.height)
  requpdate()
}

function genpersonattrs() {
}

function genrandomattrs() {
  document.getElementById('tag0in').value = '';
  document.getElementById('tag1in').value = '';

  for (var i = 0; i < 40; ++i) {
    var cb = document.getElementById("attr" + i)
    var r = Math.floor(Math.random() * 256)
    cb.curval = r;
    cb.style.backgroundColor = mkcol([r,r,r])
  }
  requpdate()
}

function mkcol(data) {
  var col = 'rgb(' + Math.floor(data[0]) + ',' + Math.floor(data[1]) + ',' + Math.floor(data[2]) + ')';
  return col
}

function makepalette(l, perm) {
  var pal = document.getElementById('palette');
  var palctx = pal.getContext('2d')
  var md2 = (pal.width/2)*(pal.width/2) + (pal.height/2)*(pal.height/2)
  var md = Math.sqrt(md2)

  pal.l = l
  pal.perm = perm

  for (var y = 0; y < pal.height; ++y) {
    for (var x = 0; x < pal.width; ++x) {
      var dx = (x - pal.width/2)
      var dy = (y - pal.height/2)
      var d2 = dx*dx+dy*dy
      var d = Math.sqrt(d2)

      var a = 255 * (x / pal.width);
      var b = 255 * (y / pal.height);

//      if (perm == 1) {
//        [l,a,b] = [g,b,r];
//      } else if (perm == 2) {
//        [l,a,b] = [b,r,g];
//      }

      var rgbcol = labtorgb([l,a,b]);
//      var r = l * (x / pal.width)
//      var g = l * (y / pal.height)
//      var b = l * (d / md)

      palctx.fillStyle = mkcol(rgbcol)
      palctx.fillRect(x, y, 1, 1)
    }
  }
}

function clickpalette(event) {
  var pal = document.getElementById('palette');
  var palctx = pal.getContext('2d')

  if (event.shiftKey || window.tool == 'p') {
    var x = event.offsetX, y = event.offsetY;
    var rgbdata = palctx.getImageData(x, y, 1, 1).data;
    pick_color(rgbdata[0], rgbdata[1], rgbdata[2])
  } else {
    var l = pal.l
    var perm = pal.perm
    if (l == 192) { l = 32 }
    else if (l == 32) { l = 128; }
    else { l = 192 }
    makepalette(l, perm)
//  } else {
//    var l = pal.l
//    var perm = pal.perm
//    ++perm; perm %= 3;
//    makepalette(l, perm)
  }
}

function pick_color(r,g,b) {
  window.curcol = [r,g,b]
//  for (var i = 1; i <= n_stages; i++) {
//    var cs = document.getElementById("stage" + i + "colorshower")
//    cs.style.backgroundColor = mkcol(window.curcol)
//  }
    var cs = document.getElementById("colorshower")
    cs.style.backgroundColor = mkcol(window.curcol)
}

var orangestr = ['#654321', '#fd6a02', '#ff8800']
var bluestr = ['#000077', '#0000bb', '#2222ff']
var uvhex = '#7f1ae5'

function start_drawing(canvas) {
  if (window.drawing == canvas) {
    return;
  }
  if (window.drawing) {
    stop_drawing()
  }
  if (window.tool == 'd') {
    var cs = document.getElementById("colorshower")
    cs.style.borderColor = orangestr[window.strength]
  } else if (window.tool == 'f') {
    var ft = document.getElementById("fuzztool")
    ft.style.borderColor = orangestr[window.strength]
  } else if (window.tool == 's') {
    var st = document.getElementById("sharptool")
    st.style.borderColor = orangestr[window.strength]
  } else if (window.tool == 'b') {
    var bt = document.getElementById("blurtool")
    bt.style.borderColor = orangestr[window.strength]
  }
  tgtlockon(canvas.stage);
  window.drawing = canvas;
}

function stop_drawing() {
  var c;
  c = bluestr[window.strength]

  if (window.tool == 'd') {
    var cs = document.getElementById("colorshower")
    cs.style.borderColor = c
  } else if (window.tool == 'f') {
    var ft = document.getElementById("fuzztool")
    ft.style.borderColor = c
  } else if (window.tool == 's') {
    var st = document.getElementById("sharptool")
    st.style.borderColor = c
  } else if (window.tool == 'b') {
    var bt = document.getElementById("blurtool")
    bt.style.borderColor = c
  }
  if (window.drawing) {
    window.drawing = false;
    requpdate()
  }
}

function fillbw(canvas, dim) {
  var ctx = canvas.getContext('2d')
  var w = canvas.width
  var h = canvas.height
  var i = 0
  for (var y = 0; y < h; y += dim) {
    for (var x = 0; x < w; x += dim) {
      var r = 0;
      if (i < canvas.cbuf.length) {
        r = canvas.cbuf[i]
      }
      ctx.fillStyle = mkcol([r,r,r])
      ctx.fillRect(x, y, dim, dim)
      ++i
    }
  }
}


function setup_canvas(id, dim) {
  var stageid = "stage" + id
  var canvas = document.getElementById(stageid);
  var adjcanvas = document.getElementById(stageid + "adj");
  var concanvas = document.getElementById(stageid + "con");
  var colorpicker = document.getElementById("colorpicker")
  var colorshower = document.getElementById("colorshower")

  canvas.stage = id
  concanvas.cbuf = new Float64Array(window.n_controls[id-1])

  var ctx = canvas.getContext('2d');
  var adjctx = adjcanvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  adjctx.imageSmoothingEnabled = false;

  var cdim = canvas.width
  adjctx.fillStyle = '#808080'
  adjctx.fillRect(0, 0, cdim, cdim)
  ctx.fillStyle = 'white'
  ctx.fillRect(0, 0, cdim, cdim)

  var scale = cdim / dim

  var alldata = ctx.getImageData(0, 0, cdim, cdim).data;
  canvas.origlab = new Float64Array(dim * dim * 3)
  canvas.rgbdata = new Float64Array(dim * dim * 3)
  canvas.adjlab = new Float64Array(dim * dim * 3)

  canvas.scale = scale
  canvas.dim = dim

  var ndata = dim * dim
  for (var i = 0; i < ndata; i = i + 1) {
    var off = i * 3
    var aoff = i * scale * 3;

    var ol = rgbtolab([alldata[aoff+0],alldata[aoff+1],alldata[aoff+2]])
    canvas.origlab[off+0] = ol[0]
    canvas.origlab[off+1] = ol[1]
    canvas.origlab[off+2] = ol[2]

    canvas.rgbdata[off+0] = alldata[aoff+0]
    canvas.rgbdata[off+1] = alldata[aoff+1]
    canvas.rgbdata[off+2] = alldata[aoff+2]

    canvas.adjlab[off+0] = 128
    canvas.adjlab[off+1] = 128
    canvas.adjlab[off+2] = 128

    var argb = labtorgb([canvas.adjlab[off+0],canvas.adjlab[off+1],canvas.adjlab[off+2]])
    adjctx.fillStyle = mkcol(argb)
    adjctx.fillRect((i % dim) * scale, Math.floor(i/dim) * scale, scale, scale);
  }


  var cdraw = function(px, py, col) {
    var off = py * dim * 3 + px * 3

    var ncol = [0,0,0]
    if (window.strength == 0) {
      var oldcol = [canvas.rgbdata[off+0], canvas.rgbdata[off+1], canvas.rgbdata[off+2]]
      ncol[0] = Math.round(0.03 * col[0] + 0.97 * oldcol[0])
      ncol[1] = Math.round(0.03 * col[1] + 0.97 * oldcol[1])
      ncol[2] = Math.round(0.03 * col[2] + 0.97 * oldcol[2])
    } else if (window.strength == 1) {
      var oldcol = [canvas.rgbdata[off+0], canvas.rgbdata[off+1], canvas.rgbdata[off+2]]
      ncol[0] = Math.round(0.2 * col[0] + 0.8 * oldcol[0])
      ncol[1] = Math.round(0.2 * col[1] + 0.8 * oldcol[1])
      ncol[2] = Math.round(0.2 * col[2] + 0.8 * oldcol[2])
    } else {
      ncol[0] = col[0]
      ncol[1] = col[1]
      ncol[2] = col[2]
    }

    canvas.rgbdata[off+0] = ncol[0]
    canvas.rgbdata[off+1] = ncol[1]
    canvas.rgbdata[off+2] = ncol[2]
    ctx.fillStyle = mkcol(ncol)
    ctx.fillRect(px * scale, py * scale, scale, scale);

    var curlab = rgbtolab(ncol)

    var adjlabcol = new Array(3)
    adjlabcol[0] = (curlab[0] - canvas.origlab[off+0]) / 2.0 + 128;
    adjlabcol[1] = (curlab[1] - canvas.origlab[off+1]) / 2.0 + 128;
    adjlabcol[2] = (curlab[2] - canvas.origlab[off+2]) / 2.0 + 128;

    canvas.adjlab[off+0] = adjlabcol[0]
    canvas.adjlab[off+1] = adjlabcol[1]
    canvas.adjlab[off+2] = adjlabcol[2]

    var adjcol = labtorgb(adjlabcol)
    adjctx.fillStyle = mkcol(adjcol)
    adjctx.fillRect(px * scale, py * scale, scale, scale);
  }

  var cblur = function(px, py) {
    var sum = [0,0,0]
    var ns = 0
    for (var y = py - 1; y <= py + 1; ++y) {
      if (y < 0) continue;
      if (y >= dim) break;
      for (var x = px - 1; x <= px + 1; ++x) {
        if (x < 0) continue;
        if (x >= dim) break;
        var off = y * dim * 3 + x * 3
        sum[0] += canvas.rgbdata[off + 0];
        sum[1] += canvas.rgbdata[off + 1];
        sum[2] += canvas.rgbdata[off + 2];
        ns++;
      }
    }
    sum[0] /= ns
    sum[1] /= ns
    sum[2] /= ns

    cdraw(px, py, sum)
  }

  var cfuzz = function(px, py) {
    var coloff = py * dim * 3 + px * 3
    var col = [canvas.rgbdata[coloff+0], canvas.rgbdata[coloff+1], canvas.rgbdata[coloff+2]];
    col[0] += (Math.random() - 0.5) * 32.0
    col[1] += (Math.random() - 0.5) * 8.0
    col[2] += (Math.random() - 0.5) * 8.0
    cdraw(px, py, col)
  }


  var csharp = function(px, py) {
    var coloff = py * dim * 3 + px * 3
    var col = [canvas.rgbdata[coloff+0], canvas.rgbdata[coloff+1], canvas.rgbdata[coloff+2]];

    var best = false
    var bestd = -1

    for (var y = py - 1; y <= py + 1; ++y) {
      if (y < 0) continue;
      if (y >= dim) break;
      for (var x = px - 1; x <= px + 1; ++x) {
        if (x < 0) continue;
        if (x >= dim) break;
        if (x == px && y == py) continue;
        var off = y * dim * 3 + x * 3
        var cand = [0,0,0]
        cand[0] = canvas.rgbdata[off + 0];
        cand[1] = canvas.rgbdata[off + 1];
        cand[2] = canvas.rgbdata[off + 2];
        if (!best) {
          best = cand
          bestd = (cand[0] - col[0]) * (cand[0] - col[0]) + (cand[1] - col[1]) * (cand[1] - col[1]) + (cand[2] - col[2]) * (cand[2] - col[2]);
        } else {
          var d = (cand[0] - col[0]) * (cand[0] - col[0]) + (cand[1] - col[1]) * (cand[1] - col[1]) + (cand[2] - col[2]) * (cand[2] - col[2]);
          if (d < bestd) {
            best = cand;
            bestd = d;
          }
        }
      }
    }

    cdraw(px, py, best)
  }


  canvas.addEventListener('mousedown', function(event) {
    var x = event.offsetX, y = event.offsetY;
    var px = parseInt(x / scale);
    var py = parseInt(y / scale);
    var off = py * dim * 3 + px * 3

    if (window.tool == "p") {
      pick_color(canvas.rgbdata[off+0], canvas.rgbdata[off+1], canvas.rgbdata[off+2])
    } else {
      start_drawing(canvas)
      if (window.tool == 'd') {
        cdraw(px, py, window.curcol);
      } else if (window.tool == 'b') {
        cblur(px, py)
      } else if (window.tool == 's') {
        csharp(px, py)
      } else if (window.tool == 'f') {
        cfuzz(px, py)
      }
    }
  })
  canvas.addEventListener('mousemove', function(event) {
    if (!window.drawing)
      return;
    if (window.drawing != canvas)
      return;

    var x = event.offsetX, y = event.offsetY;
    var px = parseInt(x / scale);
    var py = parseInt(y / scale);

    if (window.tool == 'd') {
      cdraw(px, py, window.curcol)
    } else if (window.tool == 'b') {
      cblur(px, py)
    } else if (window.tool == 's') {
      csharp(px, py)
    } else if (window.tool == 'f') {
      cfuzz(px, py)
    }
  })



}


function doubleclick(id, cb, targetdivid) {
          var div = document.getElementById(id)
  var targetdiv;
  if (targetdivid) {
    targetdiv = document.getElementById(targetdivid)
  }

          if (div.nclicks == 1) {
            div.nclicks = 2;
            div.style.borderColor='#00ff00'
            if (targetdiv)
              targetdiv.style.borderColor=orangestr[1]
            setTimeout(function() { div.nclicks = 0; div.style.borderColor='#606010'; if (targetdiv) { targetdiv.style.borderColor = 'gray' } }, 400)
            cb()
          } else if (div.nclicks == 2) {
            return;
          } else if (div.nclicks == -1) {
            return;
          } else { // 0
            div.nclicks = 1;
            div.style.borderColor='yellow'
//            if (targetdiv)
//              targetdiv.style.borderColor='yellow'

            setTimeout(function() {
              if (div.nclicks != 1) return;
              div.style.borderColor='red';
              if (targetdiv)
                targetdiv.style.borderColor='gray'
              div.nclicks = -1

              setTimeout(function() {
                div.nclicks = 0
                div.style.borderColor='#606010'
              }, 400)
            }, 400)
          }
}

function doburnin(nclicks) {
  var hyper = new Uint8Array(8);
  hyper.fill(0);
  for (var i = 0; i < 8; ++i) {
    hyper[i] = 16 * nclicks
  }
return;
//  requpdate(hyper)

  var border = document.getElementById("burnin")
  border.style.borderColor = orangestr[1]
}

function doubleclick2(id, cb) {
          var div = document.getElementById(id)

          if (div.nclicks == 1) {
            div.nclicks = 2;
            div.style.borderColor=orangestr[2]
            setTimeout(function() { div.nclicks = 0; div.style.borderColor='gray'; }, 400)
            cb(2)
            return;
          } else if (div.nclicks == 2) {
            div.nclicks = 3;
            div.style.borderColor='white'
            setTimeout(function() { div.nclicks = 0; div.style.borderColor='gray'; }, 400)
            cb(3)
            return;
          } else if (div.nclicks == 3) {

          } else { // 0 too
            div.nclicks = 1;
            div.style.borderColor=orangestr[1]

            cb(1)

            setTimeout(function() {
              if (div.nclicks != 1) return;
              div.style.borderColor='gray';
              div.nclicks = 0
            }, 400)
          }
}

//  https://stackoverflow.com/a/33703102
function concatTypedArrays(a, b) { // a, b TypedArray of same type
    var c = new (a.constructor)(a.length + b.length);
    c.set(a, 0);
    c.set(b, a.length);
    return c;
}

function readTypedArray(a, n) { //
    var newa = new (a.constructor)(a.length - n);
    var c = new (a.constructor)(n)
    c.set(a.slice(0, n));
    newa.set(a.slice(n))
    return [newa, c];
}


// https://jsfiddle.net/ssell/qzzvruc4/
function gaussianRand(rng) {
      var x1 = 0.0;
      var x2 = 0.0;
      var w  = 0.0;

      do {
          // Math.random() gives value on range [0, 1) but
          // the Polar Form expects [-1, 1].
          x1 = (2.0 * rng()) - 1.0;
          x2 = (2.0 * rng()) - 1.0;
          w  = (x1 * x1) + (x2 * x2);
      } while(w >= 1.0);

      w = Math.sqrt((-2.0 * Math.log(w)) / w);

      return x1 * w;
}

// https://stackoverflow.com/questions/521295/seeding-the-random-number-generator-in-javascript
function mulberry32(a) {
    return function() {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}


// https://stackoverflow.com/a/21394730
function sendMessage(sock, msg){
    // Wait until the state of the socket is not ready and send the message when it is...
    waitForSocketConnection(sock, function(){
//alert('sending msg')
        sock.send(msg);
    });
}

// Make the function wait until the connection is made...
function waitForSocketConnection(socket, callback){
    setTimeout(
        function () {
            if (socket.readyState === 1) {
                console.log("connected")
                callback();
                return;

            } else {
                console.log("connecting")
                waitForSocketConnection(socket, callback);
            }
        }, 200);
}

function requpdatecon(stage, hcmd) {
  var hyper = new Uint8Array(8);
  hyper.fill(0);
  hyper[0] = hcmd
  hyper[1] = (1 << (stage - 1));
  requpdate(hyper)

  var border = document.getElementById("stage" + stage + "conborder");
  border.style.borderColor = orangestr[1]
}

function updatecon(newcon) {
  var tattrs = 40
  var reqoff = 0 + tattrs;

  for (var stage = 1; stage <= 4; ++stage) {
    var border = document.getElementById("stage" + stage + "conborder");
    border.style.borderColor = 'gray'
  }
  var border = document.getElementById("burnin")
  border.style.borderColor = 'gray'

  var j = 0
  for (var i = 0; i < 4; i++) {
    var stagecon = document.getElementById('stage' + (i + 1) + "con")
    var nc = window.n_controls[i]
    for (var c = 0; c < nc; ++c) {
      stagecon.cbuf[c] = newcon[j]
      ++j
    }
    fillbw(stagecon, 16)
  }
  if (j != window.tcontrols)  {
    alert('error' + j + ' ' + window.tcontrols)
  }
}

function reqgen() {
  var tattrs = 40
  var tcontrols = window.tcontrols;
  var nbuf = namebuf()

  var reqsize = 8 + nbuf.length

  var reqbuf = new Uint8Array(reqsize)
  var reqoff = 0

  reqbuf[0] = 0
  for (var i = 1; i < 8; ++i) reqbuf[i] = 0;
  reqoff += 8;

  reqbuf.set(nbuf, reqoff)
  reqoff += nbuf.length

  if (reqoff != reqsize) {
    alert('huh')
  }

  sendMessage(window.socket, reqbuf)
}




function requpdate(hyper) {
  if (!window.allready)
    return;

  var tattrs = 40
  var tadjust = 0
  var tcontrols = 0
  for (var i = 0; i < 4; i++) {
    tadjust += n_adjust[i]
    tcontrols += window.n_controls[i]
  }
  var ttarget = 64*64*3
  var nbuf = namebuf()
  var thyper = 8

  var reqsize = 8 + nbuf.length + thyper + tattrs + 8*tcontrols + 8*tadjust + 32
  var reqoff = 0

  var reqbuf = new Uint8Array(reqsize)
  reqbuf[0] = 1
  for (var i = 1; i < 8; ++i) {
    reqbuf[i] = 0
  }
  reqbuf[1] = window.tgtlock;
  reqbuf[2] = window.ctrlock;
  reqoff += 8

  reqbuf.set(nbuf, reqoff)
  reqoff += nbuf.length

  if (hyper) {
    for (var i = 0; i < thyper; ++i) {
      reqbuf[reqoff + i] = hyper[i]
    }
  } else {
    for (var i = 0; i < thyper; ++i) {
      reqbuf[reqoff + i] = 0
    }
  }
  reqoff += thyper

  var reqattrs = new Uint8Array(reqbuf.buffer, reqoff, tattrs)
  reqoff += tattrs
  for (var i = 0; i < tattrs; ++i) {
    var at = document.getElementById('attr' + i)
    reqattrs[i] = at.curval
  }

  var reqcons = new Float64Array(reqbuf.buffer, reqoff, tcontrols)
  reqoff += tcontrols * 8
  var coff = 0
  for (var i = 0; i < 4; i++) {
    var stagecon = document.getElementById('stage' + (i + 1) + "con")
    var nc = window.n_controls[i]

    for (var c = 0; c < nc; ++c) {
      reqcons[coff + c] = stagecon.cbuf[c]
    }
    coff += nc
  }
  if (coff != tcontrols) { window.alert('what1') }

  var reqadj = new Float64Array(reqbuf.buffer, reqoff, tadjust)
  reqoff += tadjust * 8
  var soff = 0
  for (var i = 0; i < 4; i++) {
    var stage = document.getElementById('stage' + (i + 1))
    for (var j = 0, jn = stage.adjlab.length; j < jn; ++j) {
      reqadj[soff + j] = (stage.adjlab[j] - 128.0) / 128.0;
    }
    soff += stage.adjlab.length
  }
  if (soff != tadjust) { window.alert('what') }

  var reqfrens = new Uint8Array(reqbuf.buffer, reqoff, 32)
  reqoff += 32
  for (var i = 0; i < 32; ++i) {
    var f = window.mynewfren;
    var k = i % 32
    var c = f.charCodeAt(k)
    if (k == 31 || isNaN(c) || !(c > 0 && c < 128)) {
      reqfrens[i] = 0
    } else {
      reqfrens[i] = c
    }
  }

  if (reqoff != reqsize) {
    alert('huh reqoff=' + reqoff + ' reqsize=' + reqsize);
  }

  //window.allready = false
  sendMessage(window.socket, reqbuf)
}

function sigmoid(x) {
  var y =  256.0/(1+Math.exp(-x/256.0));
  if (y > 255.5) {
    y = 255.5;
  } else if (y < 0.5) {
    y = 0.5;
  }
  return y;
}
function unsigmoid(y) {
  var x = y / 256.0
  return Math.log(-x/(x-1)) * 256.0;
}


function updatectx(newattrs) {
  for (var i = 0; i < 40; ++i) {
          var at1 = document.getElementById('attr' + i)
          var nval = newattrs[i];
          at1.style.backgroundColor = mkcol([nval, nval, nval])
          at1.curval = nval

  }
}


function updategen(newgendata) {
  var stage = document.getElementById('stage4');
  var stagegen = document.getElementById('stage4gen');
  var ctx = stagegen.getContext('2d')

  window.gendata = newgendata;

  for (var i = 0; i < 64*64*3; i += 3) {
    var col = labtorgb([newgendata[i+0], newgendata[i+1], newgendata[i+2]])

    var px = Math.floor(i/3) % stage.dim
    var py = Math.floor(Math.floor(i/3) / stage.dim)
    var scale = stage.scale
    ctx.fillStyle = mkcol(col)
    ctx.fillRect(px * scale, py * scale, scale, scale);
  }


  var fnewgendata = new Float64Array(32*32*3);
  for (var y = 0; y < 32; ++y) {
    for (var x = 0; x < 32; ++x) {
      for (var c = 0; c < 3; c++) {
        var i = 32*3*y + 3*x + c;
        var j = 64*3*(2*y) + 3*(2*x) + c;
        var z = (newgendata[j] + newgendata[j+3] + newgendata[j+64*3] + newgendata[j+64*3+3]) / 4.0
        fnewgendata[i] = z;
      }
    }
  }

  stage = document.getElementById('stage3');
  stagegen = document.getElementById('stage3gen');
  ctx = stagegen.getContext('2d')

  for (var i = 0; i < 32*32*3; i += 3) {
    var col = labtorgb([fnewgendata[i+0], fnewgendata[i+1], fnewgendata[i+2]])

    var px = Math.floor(i/3) % stage.dim
    var py = Math.floor(Math.floor(i/3) / stage.dim)
    var scale = stage.scale
    ctx.fillStyle = mkcol(col)
    ctx.fillRect(px * scale, py * scale, scale, scale);
  }

  var gnewgendata = new Float64Array(16*16*3);
  for (var y = 0; y < 16; ++y) {
    for (var x = 0; x < 16; ++x) {
      for (var c = 0; c < 3; c++) {
        var i = 16*3*y + 3*x + c;
        var j = 32*3*(2*y) + 3*(2*x) + c;
        var z = (fnewgendata[j] + fnewgendata[j+3] + fnewgendata[j+32*3] + fnewgendata[j+32*3+3]) / 4.0
        gnewgendata[i] = z;
      }
    }
  }

  stage = document.getElementById('stage2');
  stagegen = document.getElementById('stage2gen');
  ctx = stagegen.getContext('2d')

  for (var i = 0; i < 16*16*3; i += 3) {
    var col = labtorgb([gnewgendata[i+0], gnewgendata[i+1], gnewgendata[i+2]])

    var px = Math.floor(i/3) % stage.dim
    var py = Math.floor(Math.floor(i/3) / stage.dim)
    var scale = stage.scale
    ctx.fillStyle = mkcol(col)
    ctx.fillRect(px * scale, py * scale, scale, scale);
  }


  var hnewgendata = new Float64Array(8*8*3);
  for (var y = 0; y < 8; ++y) {
    for (var x = 0; x < 8; ++x) {
      for (var c = 0; c < 3; c++) {
        var i = 8*3*y + 3*x + c;
        var j = 16*3*(2*y) + 3*(2*x) + c;
        var z = (gnewgendata[j] + gnewgendata[j+3] + gnewgendata[j+16*3] + gnewgendata[j+16*3+3]) / 4.0
        hnewgendata[i] = z;
      }
    }
  }

  stage = document.getElementById('stage1');
  stagegen = document.getElementById('stage1gen');
  ctx = stagegen.getContext('2d')

  for (var i = 0; i < 8*8*3; i += 3) {
    var col = labtorgb([hnewgendata[i+0], hnewgendata[i+1], hnewgendata[i+2]])

    var px = Math.floor(i/3) % stage.dim
    var py = Math.floor(Math.floor(i/3) / stage.dim)
    var scale = stage.scale
    ctx.fillStyle = mkcol(col)
    ctx.fillRect(px * scale, py * scale, scale, scale);
  }

}

function updatelocks(newlocks) {
  window.tgtlock = newlocks[0]
  window.ctrlock = newlocks[1]

  for (var stage = 1; stage <= 4; ++stage) {
    var adjcanvas = document.getElementById("stage" + stage + "adjborder")
    var concanvas = document.getElementById("stage" + stage + "conborder")

    if (window.tgtlock & (1 << (stage - 1))) {
      adjcanvas.style.borderColor = 'blue'
    } else {
      adjcanvas.style.borderColor = 'gray'
    }

    if (window.ctrlock & (1 << (stage - 1))) {
      concanvas.style.borderColor = 'blue'
    } else {
      concanvas.style.borderColor = 'gray'
    }
  }
}

function updatefrens(newfrensbuf) {
  var newfrens = new Array(16)

  for (var i = 0; i < 512; ++i) {
    var j = Math.floor(i / 32);
    var f = window.frens[j];
    var c = newfrensbuf[i]
    if (i % 32 == 0) {
      newfrens[j] = ''
    }
    if (c > 0) {
      newfrens[j] = newfrens[j] + String.fromCharCode(c);
    }
  }

  for (var i = 0; i < 16; ++i) {
    if (window.frens[i] != newfrens[i] || (window.dirtyfren && newfrens[i] == window.dirtyfren)) {
      var td = document.getElementById("fren" + i)
      window.frens[i] = newfrens[i]
      var nom = window.frens[i]
      if (nom == '') {
        td.innerHTML = ""
        td.style.borderStyle = 'none'
        td.style.borderWidth = '2px'
      } else {
        var newhtml =  "<img width=100 height=100 id='image_" + nom + "' onMouseOver='mouseoverfren(" + i + ")' " +
           " onMouseOut='mouseoutfren(" + i + ")' " +
           " onClick='clickfren(" + i + ")' src='profile.cgi/" + nom;
        if (newfrens[i] == window.dirtyfren) {
          newhtml += "?nonce=" + Math.floor(Math.random() * 999999);
        }
        newhtml += "'><br/>" + nom
        td.innerHTML = newhtml
        td.style.borderStyle = 'solid'
        td.style.borderColor = 'gray'
        td.style.borderWidth = '2px'
      }
    }
  }

  frenbufhilite()
}

function mouseoverfren(i) {
  var frentd = document.getElementById('fren' + i)
  frentd.style.borderColor =  (document.getElementById('frenbuf').value == window.frens[i] ? 'blue' : 'yellow');
}
function mouseoutfren(i) {
  var frentd = document.getElementById('fren' + i)
  frentd.style.borderColor = (document.getElementById('frenbuf').value == window.frens[i] ? 'blue' : 'gray');
}

function clickfren(i) {
  var frenbuf = document.getElementById('frenbuf')
  if (frenbuf.value == window.frens[i]) {
    gotofile()
    return
  }
  frenbuf.value = window.frens[i]
  frenbufhilite()
}

function doupdate(newlabdata, newcontextdata, newcontroldata, newadjdata, newgendata, newlocks, newfrens) {
  window.allready = true

  updatectx(newcontextdata)
  updatecon(newcontroldata)
  updategen(newgendata)
  updatelocks(newlocks)
  updatefrens(newfrens)

  var stage4 = document.getElementById('stage4')
  var stage4ctx = stage4.getContext('2d')
  var stage4adj = document.getElementById('stage4adj')
  var stage4adjctx = stage4adj.getContext('2d')
  var stage4orig = document.getElementById('stage4orig')
  var stage4origctx = stage4orig.getContext('2d')

  stage4.adjlab = new Float64Array( newadjdata.slice((8*8+16*16+32*32)*3) );
  for (var i = 0; i < 64*64*3; ++i) {
    var norig = newlabdata[i] - ((stage4.adjlab[i] - 128) * 2.0);
    stage4.origlab[i] = norig;
  }
  for (var i = 0; i < 64*64*3; i += 3) {
    var col = labtorgb([newlabdata[i+0], newlabdata[i+1], newlabdata[i+2]])
    stage4.rgbdata[i+0] = col[0]
    stage4.rgbdata[i+1] = col[1]
    stage4.rgbdata[i+2] = col[2]
    var px = Math.floor(i/3) % stage4.dim
    var py = Math.floor(Math.floor(i/3) / stage4.dim)
    var scale = stage4.scale
    stage4ctx.fillStyle = mkcol(col)
    stage4ctx.fillRect(px * scale, py * scale, scale, scale);

    var adjrgb = labtorgb([stage4.adjlab[i+0], stage4.adjlab[i+1], stage4.adjlab[i+2]]);
    stage4adjctx.fillStyle = mkcol(adjrgb)
    stage4adjctx.fillRect(px * scale, py * scale, scale, scale);

    var origrgb = labtorgb([stage4.origlab[i+0], stage4.origlab[i+1],stage4.origlab[i+2]]);
    stage4origctx.fillStyle = mkcol(origrgb)
    stage4origctx.fillRect(px * scale, py * scale, scale, scale);
  }
  fnewlabdata = new Float64Array(32*32*3);
  for (var y = 0; y < 32; ++y) {
    for (var x = 0; x < 32; ++x) {
      for (var c = 0; c < 3; c++) {
        var i = 32*3*y + 3*x + c;
        var j = 64*3*(2*y) + 3*(2*x) + c;
       var z = (stage4.origlab[j] + stage4.origlab[j+3] + stage4.origlab[j+64*3] + stage4.origlab[j+64*3+3]) / 4.0
        fnewlabdata[i] = z;
      }
    }
  }


  var stage3 = document.getElementById('stage3')
  var stage3ctx = stage3.getContext('2d')
  var stage3adj = document.getElementById('stage3adj')
  var stage3adjctx = stage3adj.getContext('2d')
  var stage3orig = document.getElementById('stage3orig')
  var stage3origctx = stage3orig.getContext('2d')

  stage3.adjlab = new Float64Array( newadjdata.slice((8*8+16*16)*3, (8*8+16*16+32*32)*3) );
  for (var i = 0; i < 32*32*3; i++) {
    var norig = fnewlabdata[i] - ((stage3.adjlab[i] - 128) * 2.0);
    stage3.origlab[i] = norig;
  }
  for (var i = 0; i < 32*32*3; i += 3) {
    var col = labtorgb([fnewlabdata[i+0], fnewlabdata[i+1], fnewlabdata[i+2]])
    stage3.rgbdata[i+0] = col[0]
    stage3.rgbdata[i+1] = col[1]
    stage3.rgbdata[i+2] = col[2]
    stage3ctx.fillStyle = mkcol(col)
    var px = Math.floor(i/3) % 32
    var py = Math.floor(Math.floor(i/3) / 32)
    var scale = stage3.scale
    stage3ctx.fillRect(px * scale, py * scale, scale, scale);

    var adjrgb = labtorgb([stage3.adjlab[i+0], stage3.adjlab[i+1], stage3.adjlab[i+2]]);
    stage3adjctx.fillStyle = mkcol(adjrgb)
    stage3adjctx.fillRect(px * scale, py * scale, scale, scale);

    var origrgb = labtorgb([stage3.origlab[i+0], stage3.origlab[i+1],stage3.origlab[i+2]]);
    stage3origctx.fillStyle = mkcol(origrgb)
    stage3origctx.fillRect(px * scale, py * scale, scale, scale);
  }
  fnewlabdata = new Float64Array(16*16*3);
  for (var y = 0; y < 16; ++y) {
    for (var x = 0; x < 16; ++x) {
      for (var c = 0; c < 3; c++) {
        var i = 16*3*y + 3*x + c;
        var j = 32*3*(2*y) + 3*(2*x) + c;
        var z = (stage3.origlab[j] + stage3.origlab[j+3] + stage3.origlab[j+32*3] + stage3.origlab[j+32*3+3]) / 4.0
        fnewlabdata[i] = z
      }
    }
  }

  var stage2 = document.getElementById('stage2')
  var stage2ctx = stage2.getContext('2d')
  var stage2adj = document.getElementById('stage2adj')
  var stage2adjctx = stage2adj.getContext('2d')
  var stage2orig = document.getElementById('stage2orig')
  var stage2origctx = stage2orig.getContext('2d')

  stage2.adjlab = new Float64Array( newadjdata.slice((8*8)*3, (8*8+16*16)*3) );
  for (var i = 0; i < 16*16*3; i++) {
    var norig = fnewlabdata[i] - ((stage2.adjlab[i] - 128) * 2.0);
    stage2.origlab[i] = norig;
  }
  for (var i = 0; i < 16*16*3; i += 3) {
    var col = labtorgb([fnewlabdata[i+0], fnewlabdata[i+1], fnewlabdata[i+2]])
    stage2.rgbdata[i+0] = col[0]
    stage2.rgbdata[i+1] = col[1]
    stage2.rgbdata[i+2] = col[2]
    stage2ctx.fillStyle = mkcol(col)
    var px = Math.floor(i/3) % 16
    var py = Math.floor(Math.floor(i/3) / 16)
    var scale = stage2.scale
    stage2ctx.fillRect(px * scale, py * scale, scale, scale);

    var adjrgb = labtorgb([stage2.adjlab[i+0], stage2.adjlab[i+1], stage2.adjlab[i+2]]);
    stage2adjctx.fillStyle = mkcol(adjrgb)
    stage2adjctx.fillRect(px * scale, py * scale, scale, scale);

    var origrgb = labtorgb([stage2.origlab[i+0], stage2.origlab[i+1],stage2.origlab[i+2]]);
    stage2origctx.fillStyle = mkcol(origrgb)
    stage2origctx.fillRect(px * scale, py * scale, scale, scale);
  }
  fnewlabdata = new Float64Array(8*8*3);
  for (var y = 0; y < 8; ++y) {
    for (var x = 0; x < 8; ++x) {
      for (var c = 0; c < 3; c++) {
        var i = 8*3*y + 3*x + c;
        var j = 16*3*(2*y) + 3*(2*x) + c;
        var z = (stage2.origlab[j] + stage2.origlab[j+3] + stage2.origlab[j+16*3] + stage2.origlab[j+16*3+3]) / 4.0
        fnewlabdata[i] = z
      }
    }
  }


  var stage1 = document.getElementById('stage1')
  var stage1ctx = stage1.getContext('2d')
  var stage1adj = document.getElementById('stage1adj')
  var stage1adjctx = stage1adj.getContext('2d')

  stage1.adjlab = new Float64Array( newadjdata.slice(0, (8*8)*3) );
  for (var i = 0; i < 8*8*3; i++) {
    var norig = fnewlabdata[i] - ((stage1.adjlab[i] - 128) * 2.0);
    stage1.origlab[i] = norig;
  }
  for (var i = 0; i < 8*8*3; i += 3) {
    var col = labtorgb([fnewlabdata[i+0], fnewlabdata[i+1], fnewlabdata[i+2]])
    stage1.rgbdata[i+0] = col[0]
    stage1.rgbdata[i+1] = col[1]
    stage1.rgbdata[i+2] = col[2]
    stage1ctx.fillStyle = mkcol(col)
    var px = Math.floor(i/3) % 8 
    var py = Math.floor(Math.floor(i/3) / 8)
    var scale = stage1.scale
    stage1ctx.fillRect(px * scale, py * scale, scale, scale);

    var adjrgb = labtorgb([stage1.adjlab[i+0], stage1.adjlab[i+1], stage1.adjlab[i+2]]);
    stage1adjctx.fillStyle = mkcol(adjrgb)
    stage1adjctx.fillRect(px * scale, py * scale, scale, scale);
  }

  var pimg = document.getElementById('profile')
  pimg.src = "profile.cgi/$NAME?nonce=" + Math.floor(Math.random() * 1000000);
}


function changetool(newtool) {
  if (newtool != "s" && newtool != "b" && newtool != "d" && newtool != "p" && newtool != 'f') {
    return;
  }
  if (window.drawing) {
    stop_drawing()
  }

  var cs = document.getElementById("colorshower")
  cs.style.borderColor = 'gray'

  var cp = document.getElementById("colorpicker")
  cp.style.borderColor = 'gray'

  var ft = document.getElementById("fuzztool")
  ft.style.borderColor = 'gray'

  var bt = document.getElementById("blurtool")
  bt.style.borderColor = 'gray'

  var st = document.getElementById("sharptool")
  st.style.borderColor = 'gray'

  if (window.tool == newtool && window.tool != 'p') {
    ++window.strength;
    window.strength = window.strength % 3;
  }
  var c = bluestr[window.strength];
  if (newtool == "s") { st.style.borderColor = c; }
  else if (newtool == "d") { cs.style.borderColor = c }
  else if (newtool == "f") { ft.style.borderColor = c }
  else if (newtool == "b") { bt.style.borderColor = c }
  else if (newtool == "p") { cp.style.borderColor = c }

  window.tool = newtool
}


window.onload = function() {
  makepalette(128, 0)
  document.getElementById('frenbuf').value = gennom()
  window.frens = new Array(16);
  window.mynewfren = ''
  for (var i = 0; i < 16; ++i) {
    window.frens[i] = '';
  }



  window.n_controls = [128, 256, 512, 1024];
  window.n_adjust = [8*8*3, 16*16*3, 32*32*3, 64*64*3]
  window.tadjust = 0;
  window.tcontrols = 0
  for (var i = 0; i < 4; ++i) {
    window.tadjust += window.n_adjust[i]
    window.tcontrols += window.n_controls[i]
  }

  var part1 = 8*(64*64*3)
  var part2 = 40
  var part3 = window.tcontrols * 8
  var part4 = window.tadjust * 8
  var part5 = 8*(64*64*3)
  var part6 = 8
  var part7 = 512
  
  var packet = part1 + part2 + part3 + part4 + part5 + part6 + part7

  window.socket = new WebSocket('ws://127.0.0.1:9999', ['binary']);
  window.socket.binaryType = 'arraybuffer';
  window.socket.inbuffer = new Uint8Array(0)
  window.socket.onmessage = function(m) {
    var md = m.data;
    var ar = new Uint8Array(md)
    window.socket.inbuffer = concatTypedArrays(window.socket.inbuffer, ar)

    while (window.socket.inbuffer.length >= packet * 2) {
      var ret = readTypedArray(window.socket.inbuffer, packet)
      window.socket.inbuffer = ret[0]
    }

    if (window.socket.inbuffer.length >= packet) {
      var buf = window.socket.inbuffer

      var labdata = new Float64Array(buf.buffer, 0, part1/8)
      var newctxdata = new Uint8Array(buf.buffer, part1, part2)
      var newctrldata = new Float64Array(buf.buffer, part1+part2, part3/8)
      var newadjdata = new Float64Array(buf.buffer, part1+part2+part3, part4/8)
      var newgendata =  new Float64Array(buf.buffer, part1+part2+part3+part4, part5/8)
      var newlocks = new Uint8Array(buf.buffer, part1+part2+part3+part4+part5, part6)
      var newfrens = new Uint8Array(buf.buffer, part1+part2+part3+part4+part5+part6, part7)

      var ret = readTypedArray(window.socket.inbuffer, packet)
      window.socket.inbuffer = ret[0]

for (var i = 0; i < labdata.length; ++i) { labdata[i] = 256 * labdata[i]; }
for (var i = 0; i < newctrldata.length; ++i) { newctrldata[i] = 256 * newctrldata[i]; }
for (var i = 0; i < newadjdata.length; ++i) { newadjdata[i] = 128 + 128 * newadjdata[i]; }
for (var i = 0; i < newgendata.length; ++i) { newgendata[i] = 256 * newgendata[i]; }

      doupdate(labdata, newctxdata, newctrldata, newadjdata, newgendata, newlocks, newfrens)
    }
  }

  window.strength = 2

  var f = function(i) { 
    var at0 = document.getElementById('attr' + i)
    at0.curval = 0
    at0.addEventListener('click',
      function(event) {
          var at1 = document.getElementById('attr' + i)
          var curval = at1.curval;

          if (event.shiftKey) { pick_color(curval, curval, curval); return }

          var nval;
          if (curval >= 0 && curval < 127) { nval = 127; }
          else if (curval >= 127 && curval < 255) { nval = 255; }
          else if (curval == 255) { nval = 0; }

          at1.style.backgroundColor = mkcol([nval, nval, nval])
          at1.style.borderColor = orangestr[1]
          at1.curval = nval
          setTimeout(function() { at1.style.borderColor = 'gray' }, 400)

          requpdate()
      }
    )
  };
  for (var i = 0; i < 40; i = i + 1) {
    f(i)
  }



  var g = function(i) { 
    var tg0 = document.getElementById('tag' + i)
    tg0.curval = 0
    tg0.addEventListener('click',
      function(event) {
          var tg1 = document.getElementById('tag' + i)
          var curval = tg1.curval;

          if (event.shiftKey) { pick_color(curval, curval, curval); return }

          var nval;
          if (curval >= 0 && curval < 128) { nval = 128; }
          else if (curval >= 128 && curval < 255) { nval = 255; }
          else if (curval == 255) { nval = 0; }

          tg1.style.backgroundColor = mkcol([nval, nval, nval])
          tg1.style.borderColor = orangestr[1]
          tg1.curval = nval
          setTimeout(function() { tg1.style.borderColor = 'gray' }, 400)

          // requpdate()
      }
    )
  };
  for (var i = 0; i < 2; i = i + 1) {
    g(i)
  }
  //genrandomattrs()

  changetool('d')

  setup_canvas(1, 8)
  setup_canvas(2, 16)
  setup_canvas(3, 32)
  setup_canvas(4, 64)

  pick_color(0,0,0)

  window.prevtool = false;

  window.addEventListener('keydown', function(event) {
    if (event.shiftKey) {
      window.prevtool = window.tool;
      changetool("p")
    } else {
      changetool(event.key)
    }
  })

  window.addEventListener('keyup', function(event) {
    if (!event.shiftKey && window.tool == "p") {
      if (window.prevtool) {
        changetool(window.prevtool);
        window.prevtool = false;
      } else {
        changetool('d')
      }
    }
  })

  window.addEventListener('mouseup', function(event) {
    stop_drawing()
  })



  var imageLoader = document.getElementById('imageLoader');
  imageLoader.addEventListener('change', handleImage, false);

//  window.allready = true;
  reqgen()
}

</script>

</html>
