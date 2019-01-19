package FunnyName;
use base Exporter;
@EXPORT = qw(genname);

sub genname {
  my @pre1 = qw(
    jay crys jun holl harv thund bowl haz zinc surf
    dun jol mar fren frod froom
    grun herk jark kev lem mel norm
    ned rugg frap sort trop vern wum
    jon joel narf dan ban trun yun
    rob cram prat
  );
  
  my @msuf1 = qw( io ius ader art on son y);
  my @fsuf1 = qw( ia ina ona adra iona i);
  
  my @pre2 = qw(
    frep mup tran string synth
    dun sami gork free blar gan net
    fron long tarp lor norm neur blur
    nimble luck just fair pool
  );
  
  my @suf2 = qw(
    bean lurva stone barn tree
    rock screen fake null grond bun
    nam plain real sticker picker
    packer macker maker bane son
    wood bull ster
    friend raver rover mark hawk razor weed walker
  );
    
  sub pick {
    $_[int rand int @_]
  }
  
  $| = 1;
  
  
  {
    my $pre1 = pick(@pre1);
    my $suf1;
    my $gender = pick('m', 'f');
    if ($gender eq 'm')  {
      $suf1 = pick(@msuf1);
    } else {
      $suf1 = pick(@fsuf1);
    }
    
    my $pre2 = pick(@pre2);
    if (rand(1) < 0.05) {
      $pre2 = "mc$pre2";
    } elsif (rand(1) < 0.01) {
      $pre2 = "van_$pre2";
    }
    my $suf2 = pick(@suf2);
    
    my $nom =  "$pre1$suf1" . "_" . "$pre2$suf2";
    wantarray ? ($nom, $gender) : $nom
  }
}

1
