#!/usr/bin/perl

use strict;

# please set
my $site_name = "UAU_baseline";

my ($arff, $pred, $out, $lab_index) = @ARGV;
if (!$arff || !$pred || !$out || !$lab_index) {
    print "Usage: $0 <arff> <pred> <out_arff> <lab-index>\n";
    exit 1;
}

my $include_frame_index = 0;

my $ai = 0;
my $class_list = "";
open(ARFF, $arff) or die "$arff: $!";
while (<ARFF>) {
    if (/\@attribute\s+(\S+)\s+(\S+)/) {
        ++$ai;
        if ($ai == $lab_index) {
            $class_list = $2;
        }
    }
    if (/\@data/) { last; }
}
close(ARFF);
if (!$class_list) {
    die "Class attribute #$lab_index not found in $arff!";
}
$class_list =~ s/^\s*\{\s*//;
$class_list =~ s/\s*\}\s*$//;
my @classes = split(",", $class_list);
map { s/^\s+//; s/\s+$//; } @classes;

open(ARFF, $arff) or die "$arff: $!";
open(PRED, $pred) or die "$pred: $!";
open(OUT, '>', $out) or die "$out: $!";
print OUT "\@relation ComParE2018_Heartbeat_Predictions_$site_name\n";
print OUT "\@attribute instance_name string\n";
#print OUT "\@attribute frame_index numeric\n";
print OUT "\@attribute prediction { ", join(", ", @classes), " }\n";

# Create a hash that maps Weka's class name abbreviations in the pred. file
# to the nominal values in the class list
my %pred2class;
my $ci = 1;
for my $class (@classes) {
    print OUT "\@attribute score_$class numeric\n";
    $pred2class{substr($class, 0, 8 - int($ci / 10))} = $class;
    ++$ci;
}
print OUT "\@data\n";


my $data = 0;
my $npred = 0;
while (<ARFF>) {
    if (/\@data/) {
        $data = 1;
    }
    elsif ($data && !/^\s*$/) {
        my @els = split(/,/);
        my $inst = $els[0];
        my $fi = $els[1];
        if (eof(PRED)) {
            print "ERROR: Wrong number of lines in $pred!\n";
            exit -1;
        }
        my $ok = 0;
        while (!eof(PRED)) {
            my $line = <PRED>;
            chomp($line);
            $line =~ s/^\s+//;
            my @els = split(/\s+/, $line);
            if ($els[0] =~ /^\d+/) {
                my @scores = split(/,/, $els[$#els]);
                map { s/\*// } @scores;
                my (undef, $pred) = split(':', $els[2]);
                if (defined $pred2class{$pred}) {
                    $pred = $pred2class{$pred};
                }
                else {
                    die "Cannot map $pred to class!\n";
                }
                print OUT $inst;
                if ($include_frame_index) {
                    print OUT ",$fi";
                }
                print OUT ",$pred,", join(",", @scores), "\n";
                $ok = 1;
                ++$npred;
                last;
            }
        }
        #last if ($npred == 100);
        if (!$ok) {
            print "ERROR: No prediction found for $inst in $pred!\n";
            exit -1;
        }
    }
}
#print "npred = $npred\n";

close(ARFF);
close(PRED);
close(OUT);
