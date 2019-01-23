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
  print "Location: http://$ENV{SERVER_NAME}/edit/$NAME\r\n\r\n";
  exit 0;
}

my $addr = unpack('N', pack('C4', split /\./, $ENV{REMOTE_ADDR}));

print "Content-type: text/html; charset=utf-8\r\n";
print "\r\n";

my %sub = (
  'README'	=> $README,
  'NAME'	=> $NAME,
  'NONCE'	=> $NONCE,
  'IPADDR'	=> $addr
);

while (<DATA>) {
  s/\$([A-Z]+)/$sub{$1}/eg;
  print;
}

__DATA__
<html> <head> <title>makemore peaple v0.2</title>
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

.colattrclass {
  padding-top: 1px;
  padding-left: 3px;
  font-family: courier;
  font-size: small
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

<table width=1500 cellpadding=4 cellspacing=0 border=3 bordercolor=gray>
  <tr>
    <td align=left valign=center style='border: 0; font-size: xx-large'>
    <b>edit parson</b>
    </td>

    <td align=right valign=top style='border: 0'><b><a href="https://github.com/jdb19937/makemore">makemore</a> peaple v0.2</b><br/></td>
  </tr>
</table>


<table width=1500 cellpadding=4 cellspacing=0 border=0>
  <tr>
  <td width=512>
    <img width=512 height=512 id="profile" src="image/$NAME.jpg?nonce=$NONCE" style="image-rendering: pixelated">
  </td>

  <td style='background-color: lightgray; border: 0' valign="top">

    <table width=970 height=320 style="border:0; overflow: hidden" cellspacing=0>
      <tr>
      <td colspan=4 valign=top height=160px>

        <table cellpadding=5 style='font-size: large'>
        <tr><td width=100px align=right><b>nom</b></td><td>$NAME</td></tr>
        <tr><td width=100px align=right><b>creator</b></td><td id=creator></td></tr>
        <tr><td width=100px align=right><b>created</b></td><td id=created></td></tr>
        <tr><td width=100px align=right><b>revisor</b></td><td id=revisor></td></tr>
        <tr><td width=100px align=right><b>revised</b></td><td id=revised></td></tr>
        </table>
      </td>
      <td align=right colspan=4 valign=top>

        <table>
        <tr>
          <td colspan=4 align=right><b style='font-size: x-large'>fam</b></td>
        </tr><tr>

          <td><span style="font-size: 100px">(</span></td>
          <td id=paren0 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
          <td id=paren1 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
          <td><span style="font-size: 100px">)</span></td>
        </tr>
        </table>

      </td>
      </tr>

      <tr>
      <td colspan=4>
      </td>
      <td colspan=4 rowspan=2 style='padding: 15px' valign=top>

        <table>
          <tr>
            <td id=fam0 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
            <td id=fam1 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
            <td id=fam2 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
          </tr>
          <tr>
            <td id=fam3 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
            <td id=fam4 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
            <td id=fam5 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
          </tr>
          <tr>
            <td id=fam6 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
            <td id=fam7 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
            <td id=fam8 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
          </tr>
        </table>
      </td>
      </tr>



      <tr>
      <td colspan=4>

        <hr/>
        <table cellpadding=5 style='font-size: large'>
          <tr>
            <td width=100px align=left colspan=2>
              <b style='font-size: x-large'>interview</b><br/><br/>
          <input id=ivbuf type=text maxlength=64 size=64 value="hello"> <input type=button value="submit" onClick="document.getElementById('ivbuf').value=''">
          <br/>
          </td></tr>

          <tr>
          <td colspan=2>
            <table>
<!-- style='border: 1px solid black' bgcolor="white" width=600 height=200 -->
            <tr><td>
              <textarea rows=4 cols=80 id="ivresponse">[response/animation should be synthesized by makemore using attrs, controls, tude, partrait, frens and fam, partraits and attrs of frens and fam, current camera image]</textarea>
            </td></tr>
            </table>
          </td>
          </tr>


          <tr>
            <td colspan=2>
              <input type=button value="enable animation" onClick="alert('unimplemented')">
              <input type=button value="enable camera" onClick="alert('unimplemented')">
              <input type=button value="burn new response" onClick="alert('unimplemented'")>
            </td>
          </tr>

<tr>
<td colspan=2><table><tr>
<hr/>
<b style='font-size: x-large'>tude</b>
<br/>


<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='persattr0' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #introvert_extrovert  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='persattr1' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #sensitive_intuitive  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='persattr6' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #polite_profane  </td></tr></table>   </td>
</tr><tr>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='persattr2' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #feeling_thinking  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='persattr3' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #perceptive_judgemental  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='persattr5' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #modest_amorous  </td></tr></table>   </td>
</tr>

<tr>
</tr>



</table></td>

</tr>

        </table>
      </td>
      </tr>
    </table>

  </td>
  </tr>
</table>
<br/>



    <table>
      <tr><td colspan=8><hr/><font size=+2><b>frens of $NAME</b></font></td></tr>

      <tr valign=top height=110>
      <td width=100 id=fren0 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren1 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren2 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren3 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren4 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren5 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren6 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren7 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      </tr>

      <tr valign=top height=110>
      <td width=100 id=fren8 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren9 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren10 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren11 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren12 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren13 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren14 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      <td width=100 id=fren15 style="border-style: none; border-width: 2px; white-space: nowrap; text-overflow:ellipsis; overflow: hidden; max-width: 100px; min-width: 100px; font-size: x-small"> </td>
      </tr>
    </table>

<br/>

<tr>

  <td align=right><b>frenbuf</b></td>
  <td colspan=8>
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





<table style='border-width: 3px; border-color: gray; border-style: solid' width=1500 cellpadding=0 cellspacing=0><tr><td>

<div id="attrcon">

<table cellpadding=0 cellspacing=1><tr>

<tr>



<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr19' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #high_cheekbones  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr29' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #rosy_cheeks  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr25' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #oval_face  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr26' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #pale_skin  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr24' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #shaven  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr0' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #stubble  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr30' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #sideburns  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr16' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #goatee  </td></tr></table>   </td>

<td onMouseOver="document.getElementById('colattrhint').style.visibility = 'visible'" onMouseOut="document.getElementById('colattrhint').style.visibility = 'hidden'"> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='colattr0' style='border: 3px solid gray; background-color: black; width: 16; height: 16'></div> </td><td valign=center class='colattrclass'> #edge_mean  </td></tr></table>   </td>
</tr><tr>


<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr22' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #mustache  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr12' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #bushy_eyebrows  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr1' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #arched_eyebrows  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr23' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #narrow_eyes  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr28' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #balding  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr4' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #bald  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr32' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #straight_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr33' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #wavy_hair  </td></tr></table>   </td>

<td onMouseOver="document.getElementById('colattrhint').style.visibility = 'visible'" onMouseOut="document.getElementById('colattrhint').style.visibility = 'hidden'"> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='colattr1' style='border: 3px solid gray; background-color: black; width: 16; height: 16'></div> </td><td valign=center class='colattrclass'> #edge_stddev  </td></tr></table>   </td>
</tr><tr>

<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr5' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #bangs  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr8' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #black_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr9' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #blonde_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr11' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #brown_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr17' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #gray_hair  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr13' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #chubby  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr14' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #double_chin  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr21' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #mouth_open  </td></tr></table>   </td>

<td onMouseOver="document.getElementById('colattrhint').style.visibility = 'visible'" onMouseOut="document.getElementById('colattrhint').style.visibility = 'hidden'"> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='colattr2' style='border: 3px solid gray; background-color: black; width: 16; height: 16'></div> </td><td valign=center class='colattrclass'> #center_mean  </td></tr></table>   </td>
</tr><tr>

<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr20' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #male  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr39' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #young  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr10' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #blurry  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr2' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #attractive  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr3' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #tired  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr6' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #big_lips  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr7' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #big_nose  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr27' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #pointy_nose  </td></tr></table>   </td>

<td onMouseOver="document.getElementById('colattrhint').style.visibility = 'visible'" onMouseOut="document.getElementById('colattrhint').style.visibility = 'hidden'"> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='colattr3' style='border: 3px solid gray; background-color: black; width: 16; height: 16'></div> </td><td valign=center class='colattrclass'> #center_stddev  </td></tr></table>   </td>

</tr><tr>


<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr31' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #smiling  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr15' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #eyeglasses  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr18' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #makeup  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr34' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #earrings  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr35' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #hat  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr36' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #lipstick  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr37' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #necklace  </td></tr></table>   </td>
<td> <table cellpadding=0 cellspacing=1 ><tr><td> <div id='attr38' style='border: 3px solid gray; background-color: gray; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #necktie  </td></tr></table>   </td>

<td> <font size="-1" color=#333333><div id="colattrhint" style="visibility: hidden">[pick color for above attributes]</font> </td>

</tr><tr>

<tr><td colspan=8 border=1></td></tr>

<script>
function vis(id) { document.getElementById(id).style.visibility = 'visible' }
function unvis(id) { document.getElementById(id).style.visibility = 'hidden' }
</script>

<td colspan=2> <table cellpadding=0 cellspacing=1 onMouseOver="vis('taghint')" onMouseOut="unvis('taghint')" ><tr><td> <div id='tag0' style='border: 3px solid gray; background-color: black; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #<input id="tag0in" size="24" maxlength=24 value="" />  </td></tr></table>   </td>

<td colspan=2> <table cellpadding=0 cellspacing=1 onMouseOver="vis('taghint')" onMouseOut="unvis('taghint')"><tr><td> <div id='tag1' style='border: 3px solid gray; background-color: black; width: 16; height: 16'></div> </td><td valign=center class='attrclass'> #<input id="tag1in" size="24" maxlength=24 value="" />  </td></tr></table>   </td>
<td id=taghint colspan=4 style='visibility: hidden'> <font color="#444444" size="-1">[enter any two tags] </font></td>

</table>


</div>

</td>
<td valign=top align=right>

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




<table cellpadding=0 cellspacing=1 width=1500 style='border-width: 3px; border-color: gray; border-style: solid'>
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
  <input type="button" for="imageLoader" value="file target" onClick="getElementById('imageLoader').click()"/>
<input type="file" size="60" id="imageLoader" name="imageLoader" accept="image/png, image/jpeg"/>


    <input type="button" for="imageLoaderCamera" value="camera target" onClick="alert('unimplemented')"/>
  <!--
    <input type="button" for="imageLoaderCamera" value="camera target" onClick="getElementById('imageLoaderCamera').click()"/>
    <input type="file" size="60" id="imageLoaderCamera" capture="user" name="imageLoaderCamera" accept="image/png, image/jpeg"/>
  <input type="button" for="imageLoaderCamera" value="camera target" onClick="loadcameraimage()"/>
  -->

  <input type="button" value="toggle legend" onClick="document.cookie = 'seenlegend=1'; var z = document.getElementById('hintlayer'); if (z.style.zIndex > 0) { z.style.zIndex = -10; z.style.visibility = 'hidden' } else { z.style.zIndex = 10; z.style.visibility = 'visible' }">
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






<table width=1500 cellpadding=0 cellspacing=4>

  <tr>
  <th style='font-size: small'>partrait</th>
  <th style='font-size: small'>palette | pretarget</th>
  <th style='font-size: small'>target</th>
  <th style='font-size: small'>adjustment</th>
  <th style='font-size: small'>controls</th>
  <th style='font-size: small'>functions</th>
  <th></th>

  <tr>

 <td width=320 style="border-width: 3px; border-color: gray; border-style: solid; padding: 0px">

<div id="hintlayer" style="position: absolute; width: 1500; height: 1500; z-index: 10" onClick="document.cookie='seenlegend=1'; this.style.visibility = 'hidden'; this.style.zIndex = -10"> <canvas id=hintcanvas width=1500 height=1500></canvas></div>

<canvas id="stage1gen" width=320 height=320></canvas></td>
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


<table width=1500 bgcolor=white cellpadding=16><tr><td>
<h1>instructions for makemore peaple v0.2</h1>
<pre style="font-size: 24px">
$README
</pre>

<form action="note.txt" method="get">
<textarea name="text" cols=80 rows=5/></textarea><br/>
<input type=submit value="send me a brief note">
</form>
</font>
</td></tR></table>


</body>

<script src="gennom.js"></script>

 <script>


function frenbufhilite() {
  var nom = document.getElementById('frenbuf').value
  for (var i = 0; i < 16; ++i) {
    var frentd = document.getElementById('fren' + i)
    frentd.style.borderColor = (nom == window.frens[i] ? 'blue' : 'gray');
  }
  for (var i = 0; i < 2; ++i) {
    var parentd = document.getElementById('paren' + i)
    parentd.style.borderColor = (nom == window.parens[i] ? 'blue' : 'gray');
  }
  for (var i = 0; i < 9; ++i) {
    var famtd = document.getElementById('fam' + i)
    famtd.style.borderColor = (nom == window.fam[i] ? 'blue' : 'gray');
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
  window.location = "edit/" + frenbuf.value
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
    img.src = "image/" + frenbuf.value + "?nonce=" + Math.floor(Math.random() * 1000000);
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

function lockalltargets(noupdate) {
  var changed = 0
  for (var stage = 1; stage <= 4; ++stage) {
    if (!(window.tgtlock & (1 << (stage - 1)))) {
      window.tgtlock |= (1 << (stage - 1));
      ++changed
    }
  }

  if (changed && !noupdate) {
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

  for (var i = 0; i < 72; ++i) {
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

function makepalette(inl, perm) {
  var pal = document.getElementById('palette');
  var palctx = pal.getContext('2d')
  var md2 = (pal.width/2)*(pal.width/2) + (pal.height/2)*(pal.height/2)
  var md = Math.sqrt(md2)

  pal.l = inl
  pal.perm = perm

  for (var y = 0; y < pal.height; ++y) {
    for (var x = 0; x < pal.width; ++x) {
      var dx = (x - pal.width/2)
      var dy = (y - pal.height/2)
      var d2 = dx*dx+dy*dy
      var d = Math.sqrt(d2)

      var l = inl;
      var a = 255 * (x / pal.width);
      var b = 255 * (y / pal.height);

      if (l == -1) {
        l = 255 * (x / pal.width);
      } else if (l == -2) {
        l = 255 * Math.abs(0.5 - (x / pal.width));
      } else if (l == -3) {
        l = 255 * (1 - Math.abs(0.5 - (x / pal.width)));
      }

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
    var perm = pal.perm ? 1 : 0
    if (l == -3) { l = 32; }
    else if (l == 32) { l = 128; }
    else if (l == 128) { l = 192; }
    else if (l == 192 ) { l = -1; }
    else if (l == -1) { l = -2; }
    else if (l == -2) { l = -3; }
    else { l = 32; }
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
  var tattrs = 72
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
  var tattrs = 72
  var tcontrols = window.tcontrols;
  var nbuf = namebuf()

  var reqsize = 8 + nbuf.length + 4

  var reqbuf = new Uint8Array(reqsize)
  var reqoff = 0

  reqbuf[0] = 0
  for (var i = 1; i < 8; ++i) reqbuf[i] = 0;
  reqoff += 8;

  reqbuf.set(nbuf, reqoff)
  reqoff += nbuf.length

  reqbuf[reqoff+0] = ($IPADDR >> 24) & 0xFF
  reqbuf[reqoff+1] = ($IPADDR >> 16) & 0xFF
  reqbuf[reqoff+2] = ($IPADDR >> 8) & 0xFF
  reqbuf[reqoff+3] = ($IPADDR >> 0) & 0xFF
  reqoff += 4

  if (reqoff != reqsize) {
    alert('huh')
  }

  sendMessage(window.socket, reqbuf)
}




function requpdate(hyper) {
  if (!window.allready)
    return;

  var tattrs = 72
  var tadjust = 0
  var tcontrols = 0
  for (var i = 0; i < 4; i++) {
    tadjust += n_adjust[i]
    tcontrols += window.n_controls[i]
  }
  var ttarget = 64*64*3
  var nbuf = namebuf()
  var thyper = 8

  var reqsize = 8 + nbuf.length + thyper + tattrs + 8*tcontrols + 8*tadjust + 32 + 4
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
  for (var i = 0; i < 40; ++i) {
    var at = document.getElementById('attr' + i)
    reqattrs[i] = at.curval
  }
  for (var i = 0; i < 4; ++i) {
    var at = document.getElementById('colattr' + i)
    reqattrs[40 + i * 3 + 0] = at.curval[0]
    reqattrs[40 + i * 3 + 1] = at.curval[1]
    reqattrs[40 + i * 3 + 2] = at.curval[2]
  }
  for (var i = 52; i < 68; ++i) {
    reqattrs[i] = 128;
  }
  reqattrs[68] = 255;
  reqattrs[69] = 0;
  reqattrs[70] = 0;
  reqattrs[71] = 0;

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

  reqbuf[reqoff+0] = ($IPADDR >> 24) & 0xFF
  reqbuf[reqoff+1] = ($IPADDR >> 16) & 0xFF
  reqbuf[reqoff+2] = ($IPADDR >> 8) & 0xFF
  reqbuf[reqoff+3] = ($IPADDR >> 0) & 0xFF
  reqoff += 4

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
  for (var i = 0; i < 4; ++i) {
          var at1 = document.getElementById('colattr' + i)
          at1.style.backgroundColor = mkcol(labtorgb([
            newattrs[40 + i * 3 + 0],
            newattrs[40 + i * 3 + 1],
            newattrs[40 + i * 3 + 2]
          ]))
          at1.curval = [
            newattrs[40 + i * 3 + 0],
            newattrs[40 + i * 3 + 1],
            newattrs[40 + i * 3 + 2]
          ];
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

function updatefam(newfambuf) {
  if (!window.fam) {
    window.fam = new Array(9)
    for (var i = 0; i < 9; ++i)
      window.fam[i] = ''
  }
  var newfam = new Array(9)

  for (var i = 0; i < 32*9; ++i) {
    var j = Math.floor(i / 32);
    var f = window.fam[j];
    var c = newfambuf[i]
    if (i % 32 == 0) {
      newfam[j] = ''
    }
    if (c > 0) {
      newfam[j] = newfam[j] + String.fromCharCode(c);
    }
  }

  for (var i = 0; i < 9; ++i) {
    if (window.fam[i] != newfam[i]) {
      var td = document.getElementById("fam" + i)
      window.fam[i] = newfam[i]
      var nom = window.fam[i]
      if (nom == '') {
        td.innerHTML = ""
        td.style.borderStyle = 'none'
        td.style.borderWidth = '2px'
      } else {
        var newhtml =  "<img width=100 height=100 id='image_" + nom + "' onMouseOver='mouseoverfam(" + i + ")' " +
           " onMouseOut='mouseoutfam(" + i + ")' " +
           " onClick='clickfam(" + i + ")' src='image/" + nom;
        newhtml += "'><br/>" + nom
        td.innerHTML = newhtml
        td.style.borderStyle = 'solid'
        td.style.borderColor = 'gray'
      }
    }
  }
}

function updateparens(newparensbuf) {
  if (!window.parens) {
    window.parens = new Array(2)
    window.parens[0] = ''
    window.parens[1] = ''
  }
  var newparens = new Array(2)

  for (var i = 0; i < 64; ++i) {
    var j = Math.floor(i / 32);
    var f = window.parens[j];
    var c = newparensbuf[i]
    if (i % 32 == 0) {
      newparens[j] = ''
    }
    if (c > 0) {
      newparens[j] = newparens[j] + String.fromCharCode(c);
    }
  }

  for (var i = 0; i < 2; ++i) {
    if (window.parens[i] != newparens[i]) {
      var td = document.getElementById("paren" + i)
      window.parens[i] = newparens[i]
      var nom = window.parens[i]
      if (nom == '') {
        td.innerHTML = ""
        td.style.borderStyle = 'none'
        td.style.borderWidth = '2px'
      } else {
        var newhtml =  "<img width=100 height=100 id='image_" + nom + "' onMouseOver='mouseoverparen(" + i + ")' " +
           " onMouseOut='mouseoutparen(" + i + ")' " +
           " onClick='clickparen(" + i + ")' src='image/" + nom;
        newhtml += "'><br/>" + nom
        td.innerHTML = newhtml
        td.style.borderStyle = 'solid'
        td.style.borderColor = 'gray'
      }
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
           " onClick='clickfren(" + i + ")' src='image/" + nom;
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

function mouseoverfam(i) {
  var famtd = document.getElementById('fam' + i)
  famtd.style.borderColor =  (document.getElementById('frenbuf').value == window.fams[i] ? 'blue' : 'yellow');
}
function mouseoverparen(i) {
  var parentd = document.getElementById('paren' + i)
  parentd.style.borderColor =  (document.getElementById('frenbuf').value == window.parens[i] ? 'blue' : 'yellow');
}
function mouseoverfren(i) {
  var frentd = document.getElementById('fren' + i)
  frentd.style.borderColor =  (document.getElementById('frenbuf').value == window.frens[i] ? 'blue' : 'yellow');
}
function mouseoutfren(i) {
  var frentd = document.getElementById('fren' + i)
  frentd.style.borderColor = (document.getElementById('frenbuf').value == window.frens[i] ? 'blue' : 'gray');
}
function mouseoutparen(i) {
  var parentd = document.getElementById('paren' + i)
  parentd.style.borderColor = (document.getElementById('frenbuf').value == window.parens[i] ? 'blue' : 'gray');
}
function mouseoutfam(i) {
  var famtd = document.getElementById('fam' + i)
  famtd.style.borderColor = (document.getElementById('frenbuf').value == window.fams[i] ? 'blue' : 'gray');
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
function clickparen(i) {
  var frenbuf = document.getElementById('frenbuf')
  if (frenbuf.value == window.parens[i]) {
    gotofile()
    return
  }
  frenbuf.value = window.parens[i]
  frenbufhilite()
}
function clickfam(i) {
  var frenbuf = document.getElementById('frenbuf')
  if (frenbuf.value == window.fam[i]) {
    gotofile()
    return
  }
  frenbuf.value = window.fam[i]
  frenbufhilite()
}

function makeip(z) {
  var a = (z>>24) & 0xFF
  var b = (z>>16) & 0xFF
  var c = (z>>8) & 0xFF
  var d = (z>>0) & 0xFF
  var ip = a + "." + b + "." + c + "." + d
  return ip
}

function updatemeta(newmetabuf) {
  var newmetabytes = new Uint8Array(newmetabuf)
  var newmeta = new Uint32Array(newmetabytes.buffer)
  document.getElementById('creator').innerHTML = newmeta[0] ? makeip(newmeta[0]) : ''
  document.getElementById('created').innerHTML = newmeta[1] ? newmeta[1] + " (-" + (Math.floor(Date.now()/1000) - newmeta[1]) + ")" : ''
  document.getElementById('revisor').innerHTML = newmeta[2] ? makeip(newmeta[2]) : ''

  document.getElementById('revised').innerHTML = newmeta[3] ? newmeta[3] + " (-" + (Math.floor(Date.now()/1000) - newmeta[3]) + ")" : ''


//  alert(newmeta)
}


function doupdate(newlabdata, newcontextdata, newcontroldata, newadjdata, newgendata, newlocks, newfrens, newparens, newfam, newmeta) {
  updatectx(newcontextdata)
  updatecon(newcontroldata)
  updategen(newgendata)
  updatelocks(newlocks)
  updatefrens(newfrens)
  updateparens(newparens)
  updatefam(newfam)
  updatemeta(newmeta)

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
  pimg.src = "image/$NAME?nonce=" + Math.floor(Math.random() * 1000000);

  if (!window.allready) {
    for (var i = 1; i <= 4; ++i) {
      tgtlockon(i)
    }
  }
  window.allready = true
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

function drawarrow(ctx, label, rot, long) {
  ctx.fillStyle = mkcol([24,24,255])
  var off = 210
  if (long) { off = 210 + 324 }
  ctx.translate(0, -10)
  ctx.fillRect(0,0,off,20)
  ctx.translate(0, 10)
  ctx.translate(off, 0)
  ctx.beginPath()
  ctx.moveTo(0, 20)
  ctx.lineTo(20, 0)
  ctx.lineTo(0, -20)
  ctx.fill()
  ctx.fillStyle = mkcol([255,255,255])
  ctx.font = '18px fixed'
  ctx.translate(-136, 5)
  if (long) {
    ctx.translate(-170, 0)
  }
  if (rot) {
    ctx.rotate(Math.PI)
    ctx.translate(-70, 9)
  }
  ctx.fillText(label, 0, 0)
  if (rot) {
    ctx.translate(70, -9)
    ctx.rotate(Math.PI)
  }
  ctx.translate(136, -5)
}


function makehint() {
  var can = document.getElementById('hintcanvas');
  var ctx = can.getContext('2d');

  var dim = 324;
  var hdim = dim/2

  ctx.save();
  ctx.translate(162, 162 + 40)
  ctx.rotate(Math.PI/2)
  drawarrow(ctx, 'generate')

  ctx.translate(120, 0)
  drawarrow(ctx, 'generate')

  ctx.translate(120, 0)
  drawarrow(ctx, 'generate')
  ctx.restore()

  ctx.save()
  ctx.translate(162 + 56, 162 - 20)
  drawarrow(ctx, '      adjust', false, true)
  ctx.restore()

  ctx.save()
  ctx.translate(dim + 166 + 40, dim + 162 + 26)
  drawarrow(ctx, '      adjust')
  ctx.restore()

  ctx.save()
  ctx.translate(dim + 166 + 40, dim + 162 + 26 + dim + 1)
  drawarrow(ctx, '      adjust')
  ctx.restore()

  ctx.save()
  ctx.translate(dim + 166 + 40, dim + 162 + 26 + dim + 1 + dim + 1)
  drawarrow(ctx, '      adjust')
  ctx.restore()

  ctx.save()
  ctx.translate(2 * dim + 166 - 81, 0 + 162 + 67)
  ctx.rotate(3 * (Math.PI / 4))
  drawarrow(ctx, 'generate', true)
  ctx.restore()

  ctx.save()
  ctx.translate(2 * dim + 166 - 81, 0 + 162 + 67 + dim + 4)
  ctx.rotate(3 * (Math.PI / 4))
  drawarrow(ctx, 'generate', true)
  ctx.restore()

  ctx.save()
  ctx.translate(2 * dim + 166 - 81, 0 + 162 + 67 + dim + 6 + dim + 6)
  ctx.rotate(3 * (Math.PI / 4))
  drawarrow(ctx, 'generate', true)
  ctx.restore()

  ctx.font = '36px fixed'
  ctx.fillStyle = mkcol([0,255,0])
  ctx.fillRect(450, 12, 500, 50)
  ctx.fillStyle = mkcol([0,0,0])
  ctx.fillText('click to hide legend and edit', 500, 49)

  ctx.font = '24px fixed'
  ctx.fillStyle = mkcol([0,255,0])
  ctx.fillRect(666, 400, 310, 42)
  ctx.fillStyle = mkcol([0,0,0])
  ctx.fillText('click to draw in target column', 676, 430)

  ctx.font = '24px fixed'
  ctx.fillStyle = mkcol([0,255,0])
  ctx.fillRect(348, 200, 280, 42)
  ctx.fillStyle = mkcol([0,0,0])
  ctx.fillText('shift+click to pick color', 376, 229)

  ctx.font = '24px fixed'
  ctx.fillStyle = mkcol([0,255,0])
  ctx.fillRect(348, 250, 280, 42)
  ctx.fillStyle = mkcol([0,0,0])
  ctx.fillText('click to rotate palette', 389, 279)

  ctx.font = '24px fixed'
  ctx.fillStyle = mkcol([0,255,0])
  var off = 1010
  var yoff = 120;
  ctx.fillRect(off, yoff, 280, 42)
  ctx.fillStyle = mkcol([0,0,0])
  ctx.fillText('click to toggle target lock', off + 18, yoff + 29)

  ctx.save();
  ctx.translate(1374, 332)
  ctx.rotate(Math.PI/2)
  ctx.font = '24px fixed'
  ctx.fillStyle = mkcol([0,255,0])
  off = 0
  ctx.fillRect(off, 0, 280, 42)
  ctx.fillStyle = mkcol([0,0,0])
  ctx.fillText('click to toggle control lock', off + 12, 26)
  ctx.restore()

  var hl = document.getElementById('hintlayer');
  if (document.cookie) {
    hl.style.visibility = 'hidden';
    hl.style.zIndex = -10;
  } else {
    hl.style.visibility = 'visible';
    hl.style.zIndex = 10;
  }
}
  
function gotMedia(mediaStream) {
  const mediaStreamTrack = mediaStream.getVideoTracks()[0];
  const imageCapture = new ImageCapture(mediaStreamTrack);
  console.log(imageCapture);
}

function loadcameraimage() {
  imageCapture.takePhoto()
  .then(blob => {
    img.src = URL.createObjectURL(blob);
    img.onload = () => { URL.revokeObjectURL(this.src); 
      var canvas = document.getElementById('imageCanvas');
      var ctx = canvas.getContext('2d');
      ctx.drawImage(img,0,0,64,64);
      var rgbdata = ctx.getImageData(0, 0, 64, 64).data;
      applyloaded(rgbdata)
    }
  })
  .catch(error => console.error('takePhoto() error:', error));
}

window.onload = function() {
//  navigator.mediaDevices.getUserMedia({video: true})
//  .then(gotMedia)
//  .catch(error => console.error('getUserMedia() error:', error));

  //alert(makeip($IPADDR))
  makehint()
  makepalette(128, 0)
  document.getElementById('frenbuf').value = gennom()
  window.fam = new Array(9);
  for (var i = 0; i < 9; i++) window.fam[i] = ''
  window.parens = new Array(2);
  window.parens[0] = ''
  window.parens[1] = ''
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
  var part2 = 72
  var part3 = window.tcontrols * 8
  var part4 = window.tadjust * 8
  var part5 = 8*(64*64*3)
  var part6 = 8
  var part7 = 512
  var part8 = 64
  var part9 = 32 * 9
  var part10 = 16
  
  var packet = part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9 + part10

  window.socket = new WebSocket('ws://' + location.host + ':9999', ['binary']);
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
      var newparens = new Uint8Array(buf.buffer, part1+part2+part3+part4+part5+part6+part7, part8)
      var newfam = new Uint8Array(buf.buffer, part1+part2+part3+part4+part5+part6+part7+part8, part9)
      var newmeta = new Uint8Array(buf.buffer, part1+part2+part3+part4+part5+part6+part7+part8+part9, part10)

      var ret = readTypedArray(window.socket.inbuffer, packet)
      window.socket.inbuffer = ret[0]

for (var i = 0; i < labdata.length; ++i) { labdata[i] = 256 * labdata[i]; }
for (var i = 0; i < newctrldata.length; ++i) { newctrldata[i] = 256 * newctrldata[i]; }
for (var i = 0; i < newadjdata.length; ++i) { newadjdata[i] = 128 + 128 * newadjdata[i]; }
for (var i = 0; i < newgendata.length; ++i) { newgendata[i] = 256 * newgendata[i]; }

      doupdate(labdata, newctxdata, newctrldata, newadjdata, newgendata, newlocks, newfrens, newparens, newfam, newmeta)
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

  var h = function(i) { 
    var at0 = document.getElementById('colattr' + i)
    at0.curval = [0,0,0]
    at0.addEventListener('click',
      function(event) {
          var at1 = document.getElementById('colattr' + i)
          var curval = at1.curval;

          if (event.shiftKey) {
var rgbcol = labtorgb(curval)
 pick_color(rgbcol[0], rgbcol[1], rgbcol[2])
}

          var nval = rgbtolab(window.curcol);

          at1.style.backgroundColor = mkcol(window.curcol)
          at1.style.borderColor = orangestr[1]
          at1.curval = nval
          setTimeout(function() { at1.style.borderColor = 'gray' }, 400)

          requpdate()
      }
    )
  };
  for (var i = 0; i < 4; i = i + 1) {
    h(i)
  }

  var hh = function(i) { 
    var at0 = document.getElementById('persattr' + i)
    at0.curval = 127;
    at0.addEventListener('click',
      function(event) {
          var at1 = document.getElementById('persattr' + i)
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

          // requpdate()
      }
    )
  };
  for (var i = 0; i < 4; i = i + 1) {
    hh(i)
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
