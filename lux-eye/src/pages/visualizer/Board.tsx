import { useHover, useMergedRef, useMouse } from '@mantine/hooks';
import { useCallback, useEffect, useState } from 'react';
import { EnvParams, Robot, Step, Tile } from '../../episode/model';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';

interface SizeConfig {
  gutterSize: number;
  tileSize: number;
  boardSize: number;
  tilesPerSide: number;
}

interface ThemeConfig {
  minimalTheme: boolean;
}

type Config = SizeConfig & ThemeConfig;

function getSizeConfig(maxWidth: number, step: Step): SizeConfig {
  const gutterSize = 1;
  const tilesPerSide = step.board.energy.length;

  let tileSize = Math.floor(Math.sqrt(maxWidth));
  let boardSize = tileSize * tilesPerSide + gutterSize * (tilesPerSide + 1);

  while (boardSize > maxWidth) {
    tileSize--;
    boardSize -= tilesPerSide;
  }

  return {
    gutterSize,
    tileSize,
    boardSize,
    tilesPerSide,
  };
}

function tileToCanvas(sizes: SizeConfig, tile: Tile): [number, number] {
  return [
    (tile.x + 1) * sizes.gutterSize + tile.x * sizes.tileSize,
    (tile.y + 1) * sizes.gutterSize + tile.y * sizes.tileSize,
  ];
}

// function scale(value: number, relativeMin: number, relativeMax: number): number {
//   const clampedValue = Math.max(Math.min(value, relativeMax), relativeMin);
//   return (clampedValue - relativeMin) / (relativeMax - relativeMin);
// }

function drawTileBackgrounds(ctx: CanvasRenderingContext2D, config: Config, step: Step, envParams: EnvParams): void {
  const board = step.board;
  const isAlternateMatch = (step.step % envParams.max_steps_in_match) * 2 < 50;

  for (let tileY = 0; tileY < config.tilesPerSide; tileY++) {
    for (let tileX = 0; tileX < config.tilesPerSide; tileX++) {
      const [canvasX, canvasY] = tileToCanvas(config, { x: tileX, y: tileY });

      ctx.fillStyle = 'white';
      ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

      let color: string;
      if (board.tileType[tileY][tileX] == 1) {
        color = '#5B5F97';
      } else if (board.tileType[tileY][tileX] == 2) {
        color = '#2c3e50';
      } else {
        const rgb = isAlternateMatch ? 150 : 75;
        // const base = isDay ? 0.1 : 0.2;
        color = `rgba(${rgb}, ${rgb}, ${rgb}, 1)`;
      }

      ctx.fillStyle = color;
      ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

      // const lichen = board.lichen[tileY][tileX];
      // if (lichen > 0) {
      //   const team = teamStrains.get(board.strains[tileY][tileX])!;
      //   ctx.fillStyle = getTeamColor(team, 0.1 + scale(lichen, 0, 100) * 0.4);
      //   ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
      // }
    }
  }

  ctx.restore();
}

function drawRobot(
  ctx: CanvasRenderingContext2D,
  config: Config,
  robot: Robot,
  team: number,
  selectedTile: Tile | null,
): void {
  const [canvasX, canvasY] = tileToCanvas(config, robot.tile);

  const isSelected = selectedTile !== null && robot.tile.x === selectedTile.x && robot.tile.y === selectedTile.y;

  ctx.fillStyle = getTeamColor(team, 1.0);
  ctx.strokeStyle = 'black';
  ctx.lineWidth = isSelected ? 2 : 1;

  const radius = config.tileSize / 2 - 1;

  ctx.beginPath();
  ctx.arc(canvasX + config.tileSize / 2, canvasY + config.tileSize / 2, radius, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawSelectedTile(ctx: CanvasRenderingContext2D, config: Config, selectedTile: Tile): void {
  const [canvasX, canvasY] = tileToCanvas(config, selectedTile);

  ctx.fillStyle = 'black';

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY - config.gutterSize,
    config.tileSize + config.gutterSize * 2,
    config.gutterSize,
  );

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY + config.tileSize,
    config.tileSize + config.gutterSize * 2,
    config.gutterSize,
  );

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY - config.gutterSize,
    config.gutterSize,
    config.tileSize + config.gutterSize * 2,
  );

  ctx.fillRect(
    canvasX + config.tileSize,
    canvasY - config.gutterSize,
    config.gutterSize,
    config.tileSize + config.gutterSize * 2,
  );

  ctx.restore();
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  config: Config,
  step: Step,
  envParams: EnvParams,
  selectedTile: Tile | null,
): void {
  ctx.save();

  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, config.boardSize, config.boardSize);
  ctx.restore();

  drawTileBackgrounds(ctx, config, step, envParams);

  for (let i = 0; i < 2; i++) {
    for (const robot of step.teams[i].robots) {
      drawRobot(ctx, config, robot, i, selectedTile);
    }
  }

  if (selectedTile !== null) {
    drawSelectedTile(ctx, config, selectedTile);
  }
}

interface BoardProps {
  maxWidth: number;
}

export function Board({ maxWidth }: BoardProps): JSX.Element {
  const { ref: canvasMouseRef, x: mouseX, y: mouseY } = useMouse<HTMLCanvasElement>();
  const { ref: canvasHoverRef, hovered } = useHover<HTMLCanvasElement>();
  const canvasRef = useMergedRef(canvasMouseRef, canvasHoverRef);

  const episode = useStore(state => state.episode);
  const turn = useStore(state => state.turn);

  const selectedTile = useStore(state => state.selectedTile);
  const setSelectedTile = useStore(state => state.setSelectedTile);

  const minimalTheme = useStore(state => state.minimalTheme);

  const [sizeConfig, setSizeConfig] = useState<SizeConfig>({
    gutterSize: 0,
    tileSize: 0,
    boardSize: 0,
    tilesPerSide: 0,
  });

  const step = episode!.steps[turn];
  const envParams = episode!.params;

  const onMouseLeave = useCallback(() => {
    setSelectedTile(null, true);
  }, []);

  useEffect(() => {
    const newSizeConfig = getSizeConfig(maxWidth, step);
    if (
      newSizeConfig.gutterSize !== sizeConfig.gutterSize ||
      newSizeConfig.tileSize !== sizeConfig.tileSize ||
      newSizeConfig.boardSize !== sizeConfig.boardSize ||
      newSizeConfig.tilesPerSide !== sizeConfig.tilesPerSide
    ) {
      setSizeConfig(newSizeConfig);
    }
  }, [maxWidth, episode]);

  useEffect(() => {
    if (!hovered) {
      return;
    }

    for (let tileY = 0; tileY < sizeConfig.tilesPerSide; tileY++) {
      for (let tileX = 0; tileX < sizeConfig.tilesPerSide; tileX++) {
        const tile = { x: tileX, y: tileY };
        const [canvasX, canvasY] = tileToCanvas(sizeConfig, tile);

        if (
          mouseX >= canvasX &&
          mouseX < canvasX + sizeConfig.tileSize &&
          mouseY >= canvasY &&
          mouseY < canvasY + sizeConfig.tileSize
        ) {
          setSelectedTile(tile, true);
          return;
        }
      }
    }
  }, [sizeConfig, mouseX, mouseY, hovered]);

  useEffect(() => {
    if (sizeConfig.tileSize <= 0) {
      return;
    }

    const ctx = canvasMouseRef.current.getContext('2d')!;

    const config = {
      ...sizeConfig,
      minimalTheme,
    };

    drawBoard(ctx, config, step, envParams, selectedTile);
  }, [step, envParams, sizeConfig, selectedTile, minimalTheme]);

  return (
    <canvas ref={canvasRef} width={sizeConfig.boardSize} height={sizeConfig.boardSize} onMouseLeave={onMouseLeave} />
  );
}
