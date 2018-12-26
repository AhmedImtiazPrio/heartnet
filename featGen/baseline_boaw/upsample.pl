#!/usr/bin/perl

use strict;

my $in = $ARGV[0];
my $out = $ARGV[1];
my $cls = $ARGV[2];
if (!$in) {
    print "Usage: $0 <arff>\n";
    exit 1;
}
if (!$out) {
    $out = $in;
    $out =~ s/\.arff$/.upsampled.arff/;
}

open(IN, $in) or die "$in: $!";
open(OUT, '>', $out) or die "$out: $!";

my %us_facts_train = (
    "0" => 3,
    "1" => 1,
    "2" => 2
);

my %us_facts_traindevel = (
    "0" => 3,
    "1" => 1,
    "2" => 2
);

my %ctr;
my $idx;
my $data = 0;
while (<IN>) {
    my $line = $_;
    ++$idx;
    chomp;
    if (/^\@/ || /^\s*$/) {
        print OUT "$_\n";
        if (/^\@data/) {
            $data = 1;
        }
    }
    elsif ($data) {
        my @els = split(/,/);
        if ($cls eq "train"){
        	my $class = $els[$#els];
        	for my $i (1..$us_facts_train{$class}) {
        		print OUT $line;
        		++$ctr{$class};	
        	}
        } 
        elsif ($cls eq "traindevel") {
        	my $class = $els[$#els];
        	for my $i (1..$us_facts_traindevel{$class}) {
        		print OUT $line;
        		++$ctr{$class};
        	}
        }
        else {
        	print "Incorrect training partition \n";
        	exit 1;
        }
    }
}
for my $k (keys %ctr) {
    print "class $k count = $ctr{$k}\n";
}
close(IN);
close(OUT);
