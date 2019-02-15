package Morescript;

sub new {
  my ($p, $ask) = @_;
  bless { ask => \&$ask }, $p
}

sub ask {
  my $this = shift;
  $this->{ask}->(@_)
}

sub do_reqsubst {
  my ($this, $rsp, $req) = @_;
  my @rsp = split /\s+/, $rsp;
  my @req = split /\s+/, $req;

  for (@rsp) {
    if (/^\\([1-9])$/) {
      $_ = $req[$1 - 1];
    }
  }

  join ' ', @rsp
}

sub do_eval {
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

  join(' ', @out)
}

sub do_joinmerge {
  my ($this, $rsp) = @_;

  my @w = split(/\s+/, $rsp);
  my @out = ( );
  
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

  join(' ', @out)
}


sub interpret {
  my ($this, $rsp, $req) = @_;

  $rsp = $this->do_reqsubst($rsp, $req);
  $rsp = $this->do_eval($rsp);
  $rsp = $this->do_joinmerge($rsp);

  if (substr($rsp, 0, 2) eq "! ") {
    $rsp = $this->ask(substr($rsp, 2));
  }

  $rsp
}

1
