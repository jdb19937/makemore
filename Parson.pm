package Parson;

sub _ns {
  my $x = shift;
  my $i = index($x, "\0");
  substr($x, $i) = '' if $i >= 0;
  $x
}

sub _ps {
  my $x = shift;
  $x = substr($x, 0, 31) if length($x) > 31;
  $x .= "\0" x (32 - length($x));
  $x
}

sub _padto {
  my ($x, $n) = @_;
  die if length($x) > $n;
  $x .= "\0" x ($n - length($x));
  $x
}
  
sub decode {
  my ($p, $enc) = @_;
  
  my $this = bless { }, $p;

  $this->{nom} = _ns( substr($enc, 0, 32) );
  $this->{parens} = [grep $_ ne '',
    _ns( substr($enc, 32, 32) ),
    _ns( substr($enc, 64, 32) ),
  ];

  $this->{frens} = [grep $_ ne '',
    map _ns( substr($enc, 32 * (3 + $_), 32) ),
    0 .. 15
  ];

  $this->{tags} = [grep $_ ne '',
    map _ns( substr($enc, 32 * (19 + $_), 32) ),
    0 .. 7
  ];

  @$this{qw{
    created revised creator revisor visited visits
    last_activity generated target_lock control_lock
  }} = unpack('V6dVCCC2', _padto( substr($enc, 864, 40), 40) );


  $$this{control_data} = _padto(substr($enc, 864 + 40, 1920), 1920);
  $$this{target_data} = _padto(substr($enc, 864 + 40 + 1920, 12288), 12288);
  $$this{partrait_data} = _padto(substr($enc, 864 + 40 + 1920 + 12288, 12288), 12288);

  $this
}

sub encode {
  my $this = shift;

  join '', (
    _ps( $$this{nom} ),
    (map _ps( $$this{parens}->[$_] ), 0, 1),
    (map _ps( $$this{frens}->[$_] ), 0 .. 15),
    (map _ps( $$this{tags}->[$_] ), 0 .. 7),
    pack('V6dCC', @$this{qw{
      created revised creator revisor visited visits
      last_activity target_lock control_lock
    }}),
    "\0" x 6,
    $$this{control_data},
    $$this{target_data},
    $$this{partrait_data},
  )
}

1

