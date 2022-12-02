"""
Module containing utility methods for the command line interface (CLI).
"""
import click

CHECK_MARK = '✓'


# == CLICK SPECIFIC UTILS ==

def echo_info(content: str, verbose: bool = True):
    if verbose:
        click.secho(f'... {content}')


def echo_success(content: str, verbose: bool = True):
    if verbose:
        click.secho(f'[{CHECK_MARK}] {content}', fg='green')


def echo_error(content: str, verbose: bool = True):
    if verbose:
        click.secho(f'[!] {content}', fg='red')
