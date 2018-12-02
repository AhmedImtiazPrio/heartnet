#!/usr/bin/perl

use strict;

sub join_arffs 
{
    my $out_arff = pop;
    my @in_arffs = @_;
    #print join(" + ", @in_arffs), " --> ", $out_arff, "\n";
    #return;
    open(OUT, '>', $out_arff);
    my $first = 1;
    for my $in_arff (@in_arffs) {
        open(IN, $in_arff);
        my $data = 0;
        my $inst = 0;
        while (<IN>) {
            #chomp;
            if (/\@data/) {
                $data = 1;
                if ($first) {
                    print OUT;
                }
            }
            elsif ($data && !/^\s*$/) {
                print OUT;
            }
            elsif ($first) {
                print OUT;
            }
        }
        close(IN);
        $first = 0;
    }
    close(OUT);
}

if ($#ARGV < 2) {
    print "Usage: $0 <arff1> <arff2> [ <arff3> ... <arffN> ] <out_arff>\n";
    exit;
}

join_arffs(@ARGV);
