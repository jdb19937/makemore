package Morescript;

sub new {
  my ($p, $ask) = @_;
  bless { ask => \&$ask }, $p
}

sub ask {
  my $this = shift;
  $this->{ask}->(@_)
}

sub interpret {
  my ($this, $rsp) = @_;
  my @w = split /\s+/, $rsp;

  my @out;
  while (@w) {
    my $w = shift(@w);

    if ($w =~ /^(\!+)(.+$)/) {
      my $n = length($1);
      unshift(@w, $2);
      my $req = join(' ', splice(@w, 0, $n));
      push @out, split /\s+/, $this->ask($req);
    } else {
      push @out, $w;
    }
  }

  @w = @out; 
  @out = ( );
  
  while (@w) {
    my $w = shift(@w);

    if ($w eq '@') {
      last if @w < 1;
      $w = shift(@w);
      push @out, split(//, $w);
    } elsif ($w eq '%') {
      push @out, join('', @w);
      last;
    } else {
      push @out, $w;
    }
  }

  $rsp = join(' ', @out);

  if (substr($rsp, 0, 2) eq "! ") {
    $rsp = $this->ask(substr($rsp, 2));
  }

  $rsp
}

1
