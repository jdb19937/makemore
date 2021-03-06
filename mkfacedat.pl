#!/usr/bin/perl -s

use FindBin qw($Bin);
my $program = join(' ', @ARGV);

use File::Temp qw(tempdir);

if ($::attrs) {
  $::dim //= 8;
  die unless $::dim == 8;
}

my $pipe = "";
if ($::dim == 8) {
  $pipe .= "| ./labshrink 128 128 4";
} elsif ($::dim == 16) {
  $pipe .= "| ./labshrink 128 128 3";
} elsif ($::dim == 32) {
  $pipe .= "| ./labshrink 128 128 2";
} elsif ($::dim == 64) {
  $pipe .= "| ./labshrink 128 128 1";
} elsif ($::dim == 128) {

} else {
  die "need dim parameter (8, 16, 32, 64, 128)";
}

my ($chan, $laydim);
if ($::gray)  {
  $pipe .= "| ./labtogray $dim $dim";
  if ($::hifreq) {
    $pipe .= "| ./grayhifreq $dim $dim";
  }
} else {
  if ($::hifreq) {
    $pipe .= "| ./labhifreq $dim $dim";
  } else {
  }
}

select(STDOUT);
$| = 1;

my $celeba = "$Bin/celeba-dataset";
my $attrfn = "$celeba/list_attr_celeba.csv";
open(my $attrfp, "< $attrfn") or die "$0: $attrfn: $!";
scalar <$attrfp>;

my $count = 0;

for my $id ('000001' .. '202599') {
  my $attrtxt = <$attrfp>;
  $attrtxt =~ s/\r//g;
  chomp $attrtxt;
  my @attr = split /,/, $attrtxt;
  shift(@attr) eq "$id.jpg" or die "attrs don't match $id $attrtxt";
  @attr == 40 or die "bad attrs $attrtxt";
  for (@attr) { die "bad attr [$_]" unless $_ eq 1 or $_ eq -1; }
  @attr = map { $_ > 0 ? 255 : 0 } @attr;
  my $attrbytes = pack('C*', @attr);

  my $jpgfn = "$celeba/img_align_celeba/$id.jpg";
  my $cmd = "djpeg -ppm < $jpgfn |pnmcut -top 20 -bottom 197 |pnmscale -width 128 |./ppmtolab $pipe";
  my $labdata = `$cmd`;
  length($labdata) == 3 * 8 * $dim * $dim or die;

  my $labbytes = pack('C*', 
    map { $_ > 255 ? 255 : $_ < 0 ? 0 : $_ }
    map { int($_ * 256) }
    unpack('d*', $labdata)
  );

  my $labbytes8 ;
  if ($dim == 8) {
    $labbytes8 = $labbytes;
  } else {
    my $cmd = "djpeg -ppm < $jpgfn |pnmcut -top 20 -bottom 197 |pnmscale -width 128 |./ppmtolab |./labshrink 128 128 4";
    my $labdata8 = `$cmd`;
    $labbytes8 = pack('C*', 
      map { $_ > 255 ? 255 : $_ < 0 ? 0 : $_ }
      map { int($_ * 256) }
      unpack('d*', $labdata8)
    );
  }

  length($attrbytes) == 40 or die "huh";
  length($labbytes) == 3 * $dim * $dim or die "huh2";
  length($labbytes8) == 3 * 8 * 8 or die "huh3";

  my $colorstats = stats($labbytes8);
  my $tags = pack('C16', (128) x 16);
  my $extra = pack('C4', 255, 0, 0, 0);

  my $packet = $attrbytes . $colorstats . $tags . $extra;
  die unless length($packet) == 72;

  print $packet;
  if (!$::attrs) {
    print $labbytes;
  }
}

sub stats {
  my $lab = shift;
  my @lab = unpack('C*', $lab);

  my @edgesum;
  my $edgecount = 0;
  for my $y (0 .. 7) {
    for my $x (0 .. 7) {
      next unless $x == 0 || $y == 0 || $x == 7;
      for my $c (0 .. 2) {
        $edgesum[$c] += $lab[$y * 8 * 3 + $x * 3 + $c];
      }
      ++$edgecount;
    }
  }
  my @edgemean = map {$_/$edgecount} @edgesum;


  my @centersum;
  my $centercount = 0;
  for my $y (2 .. 5) {
    for my $x (3 .. 4) {
      for my $c (0 .. 2) {
        $centersum[$c] += $lab[$y * 8 * 3 + $x * 3 + $c];
      }
      ++$centercount;
    }
  }
  my @centermean = map {$_/$centercount} @centersum;

  my @mouthsum;
  my $mouthcount = 0;
  for my $y (5,6) {
    for my $x (3,4) {
      for my $c (0 .. 2) {
        $mouthsum[$c] += $lab[$y * 8 * 3 + $x * 3 + $c];
      }
      ++$mouthcount;
    }
  }
  my @mouthmean = map {$_/$mouthcount} @mouthsum;

  my @hairsum;
  my $haircount = 0;
  for my $y (1) {
    for my $x (2,3,4,5) {
      for my $c (0 .. 2) {
        $hairsum[$c] += $lab[$y * 8 * 3 + $x * 3 + $c];
      }
      ++$haircount;
    }
  }
  my @hairmean = map {$_/$haircount} @hairsum;



  my @stats = (@edgemean, @hairmean, @centermean, @mouthmean);
  $_ = int($_) for @stats;
  for (@stats) { if ($_ > 255) { $_ = 255; } if ($_ < 0) { $_ = 0; } }
  pack 'C12', @stats
}
